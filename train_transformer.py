import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import BrainToTextDataset, train_test_split_indicies
from utils import gauss_smooth

import torchaudio.functional as F # for edit distance
import torch.nn.functional as F2
from omegaconf import OmegaConf

#torch.set_float32_matmul_precision('high') # makes float32 matmuls faster on some GPUs
#torch.backends.cudnn.deterministic = True # makes training more reproducible
#torch._dynamo.config.cache_size_limit = 64

from models.model import *
from models.datasets import Brain2TextDS




class BrainToText_Trainer:
    def __init__(self, args):
        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 
        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing
        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 
        self.transform_args = self.args['dataset']['data_transforms']
        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=False)
        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=False)
        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        
        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        self.logger.info(f'Using device: {self.device}')
        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TranformerED(
            neural_dim = self.args['model']['n_input_features'],
            n_units = self.args['model']['n_units'],
            n_days = len(self.args['dataset']['sessions']),
            n_classes  = self.args['dataset']['n_classes'],
            rnn_dropout = self.args['model']['rnn_dropout'], 
            input_dropout = self.args['model']['input_network']['input_layer_dropout'], 
            n_layers = self.args['model']['n_layers'],
            patch_size = self.args['model']['patch_size'],
            patch_stride = self.args['model']['patch_stride'],
        ).to(self.device)

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

        # Call torch.compile to speed up training
        # self.logger.info("Using torch.compile")
        # self.model = torch.compile(self.model)
        self.logger.info(f"Initialized Transformer model")
        self.logger.info(self.model)
        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")
        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")
        # Create datasets and dataloaders
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]
        # Ensure that there are no duplicate days
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        val_set = Brain2TextDS(data_path = val_file_paths)
        train_set = Brain2TextDS(data_path = train_file_paths,)
        
        self.train_loader = DataLoader(train_set, batch_size = self.args['dataset']['batch_size'], shuffle = self.args['dataset']['train_loader_shuffle'], num_workers = self.args['dataset']['num_dataloader_workers'], pin_memory = True)
        self.val_loader = DataLoader(val_set, batch_size = self.args['dataset']['val_batch_size'], shuffle = self.args['dataset']['val_loader_shuffle'], num_workers = self.args['dataset']['num_dataloader_workers'], pin_memory = True)
        self.logger.info("Successfully initialized datasets")

        self.optimizer = self.create_optimizer()
        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")

        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])
        self.model.to(self.device)

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        # bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        # day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        # other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        # if len(day_params) != 0:
        #     param_groups = [
        #             {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
        #             {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
        #             {'params' : other_params, 'group_type' : 'other'}
        #         ]
        # else: 
        #     param_groups = [
        #             {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
        #             {'params' : other_params, 'group_type' : 'other'}
        #         ]
            
        # optim = torch.optim.AdamW(
        #     param_groups,
        #     lr = self.args['lr_max'],
        #     betas = (self.args['beta0'], self.args['beta1']),
        #     eps = self.args['epsilon'],
        #     weight_decay = self.args['weight_decay'],
        #     fused = True
        # )



        bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        param_groups = [
                {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                {'params' : other_params, 'group_type' : 'other'}
            ]
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )

        return optim 

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay

            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        ''' 
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only = False) # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)
        
        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        '''
        Save a training checkpoint
        '''

        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            #'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : PER,
            'val_loss' : loss
        }
        
        torch.save(checkpoint, save_path)
        
        self.logger.info("Saved model to checkpoint: " + save_path)

        # Save the args file alongside the checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def train(self):
        self.model.train()
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []
        val_steps_since_improvement = 0
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        train_start_time = time.time()
        for epoch in range(self.args['num_epochs']):
            step_times = 0
            for i, batch in enumerate(self.train_loader):
                max_len = max(batch['features_len'].max(), batch['phone_seq_lens'].max())
                # print(batch['phone_seq_lens'], batch['features_len'].max())
                # print(batch['input_features'].shape)
                batch['input_features'] = batch['input_features'][:, :max_len]

                # Move data to device
                features = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)
                tgt_in = batch['tgt_in'].to(self.device)
                tgt_in = F2.pad(tgt_in, (0, max_len - tgt_in.size(1))) if tgt_in.size(1) < max_len else tgt_in[:,:max_len]
                tgt_out = batch['tgt_out'].to(self.device)
                tgt_out = F2.pad(tgt_out, (0, max_len - tgt_out.size(1))) if tgt_out.size(1) < max_len else tgt_out[:,:max_len]
                seq_pad = batch['seq_pad'][:,:max_len].to(self.device)
                
                self.model.train()
                if step_times % self.args['lr_update_step'] == 0:
                    self.optimizer.zero_grad()

                start_time = time.time() 

                # Use autocast for efficiency
                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):

                    # Apply augmentations to the data
                    #print("transform data", features.shape, n_time_steps)
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'train')
                    #print(features.shape)

                    #adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)
                    adjusted_lens = ((n_time_steps)).to(torch.int32)

                    # Get phoneme predictions 
                    assert tgt_in.shape == seq_pad.shape
                    #print(features.shape, tgt_in.shape, seq_pad.shape, day_indicies.shape)
                    #input("train")
                    logits = self.model(features, day_indicies, tgt_in, seq_pad)

                    # Calculate CTC Loss
                    # print(labels.shape, labels[0]) # 音素ID
                    # print(logits.shape, logits[0])
                    # print(n_time_steps, adjusted_lens)
                    # print(phone_seq_lens.shape, phone_seq_lens[0])
                    # input("Ne")
                    loss = self.loss(
                        logits.reshape(-1, logits.size(-1)), 
                        tgt_out.reshape(-1)
                    )

                    train_losses.append(loss.detach().cpu().numpy())
                
                loss.backward()
                if self.args['grad_norm_clip_value'] > 0: 
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               max_norm = self.args['grad_norm_clip_value'],
                                               error_if_nonfinite = True,
                                               foreach = True
                                               )
                if step_times % self.args['lr_update_step'] == self.args['lr_update_step'] - 1:
                    self.optimizer.step()
                    self.learning_rate_scheduler.step()
                step_times += 1
                
                if i == 0:
                    self.logger.info(f'Train epoch {epoch}: ' +
                            f'loss: {(loss.detach().item()):.2f} ' +
                            f'grad norm: {grad_norm:.2f} '
                            )        
            # epoch done
            if epoch % 5 ==0:
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_losses.append(val_metrics['avg_loss'])
                val_PERs.append(val_metrics['avg_PER'])

                self.logger.info(f'Val batch {i}: ' +
                            f'PER (avg): {val_metrics["avg_PER"]:.4f} ' +
                            f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} '
                            )
                    
                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss): 
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f) 
                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement +=1

            self.logger.info(f'End of epoch {epoch}')
                
        # Log final training steps 
        training_duration = time.time() - train_start_time

        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model 
        if self.args['save_final_model']:
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses 
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = None #val_results

        return train_stats

    def validation(self, loader, return_logits = False, return_data = False):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()

        metrics = {}
        
        # Record metrics
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []


        total_edit_distance = 0
        total_seq_length = 0

        # Calculate PER for each specific day
        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in tqdm(enumerate(loader), desc = 'Validation'):    
            max_len = max(batch['features_len'].max(), batch['phone_seq_lens'].max())
            batch['input_features'] = batch['input_features'][:, :max_len]
                
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            tgt_in = batch['tgt_in'].to(self.device)
            tgt_in = F2.pad(tgt_in, (0, max_len - tgt_in.size(1))) if tgt_in.size(1) < max_len else tgt_in[:,:max_len]
            tgt_out = batch['tgt_out'].to(self.device)
            tgt_out = F2.pad(tgt_out, (0, max_len - tgt_out.size(1))) if tgt_out.size(1) < max_len else tgt_out[:,:max_len]
            seq_pad = batch['seq_pad'][:,:max_len].to(self.device)

            if self.args['dataset']['dataset_probability_val'][day_indicies[0].item()] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day_indicies[0].item()}")
                    raise ValueError(f"代码错误，不应该有未出现的天 {day_indicies[0].item()}")
                continue
            
            with torch.no_grad():
                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')
                    #adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)
                    adjusted_lens = ((n_time_steps)).to(torch.int32)

                    logits = self.model(features, day_indicies, tgt_in, seq_pad)

                    loss = self.loss(
                        logits.reshape(-1, logits.size(-1)), 
                        tgt_out.reshape(-1)
                    )

                metrics['losses'].append(loss.detach().cpu().numpy())
                
                batch_edit_distance = 0
                decoded_seqs = []
                for iterIdx, day in enumerate(day_indicies):
                    decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].detach().clone(), dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.detach().cpu().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0])
                    trueSeq = labels[iterIdx][:phone_seq_lens[iterIdx]].detach().cpu().numpy()

                    distd = F.edit_distance(decoded_seq, trueSeq) # return: int
                    batch_edit_distance += distd
                    day_per[day.item()]['total_edit_distance'] += distd
                    day_per[day.item()]['total_seq_length'] += phone_seq_lens[iterIdx].item()
                    decoded_seqs.append(decoded_seq)

            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            # Record metrics
            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].cpu().numpy())
            metrics['trial_nums'].append(batch['trial_nums'].cpu().numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_length

        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics
    
    def transform_data(self, features, n_time_steps, mode = 'train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut
                if cut > 0: features = F2.pad(features, (0, 0, 0, cut))
                

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        
        return features, n_time_steps





if __name__ == '__main__':
    from omegaconf import OmegaConf
    import argparse, shutil

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    argssss = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    args = OmegaConf.load('transformer_args.yaml')

    if argssss.debug and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)


    trainer = BrainToText_Trainer(args)
    metrics = trainer.train()

    
