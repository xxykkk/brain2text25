import torch
from torch import nn
import string, random
import transformers


from torch.utils.data import Dataset
import h5py, os
from tqdm import tqdm
import numpy as np

class Brain2TextDS(Dataset):
    def load_h5py_file(self, file_path, day):
        data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [],
            'seq_len': [],
            'transcriptions': [],
            'sentence_label': [],
            'session': [],
            'block_nums': [],
            'trial_nums': [],

            'day_indicies': [],
            'features_len': [],

            'seq_pad': [],
        }
        # Open the hdf5 file for that day
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            # print(keys)
            # input("Next")
            # For each trial in the selected trials in that day
            for key in keys:
                g = f[key]

                neural_features = g['input_features'][:]
                n_time_steps = g.attrs['n_time_steps']
                seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
                seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
                transcription = g['transcription'][:] if 'transcription' in g else None
                sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
                session = g.attrs['session']
                block_num = g.attrs['block_num']
                trial_num = g.attrs['trial_num']
                
                data['neural_features'].append(torch.tensor(neural_features, dtype=torch.float32))
                data['n_time_steps'].append(torch.tensor(n_time_steps, dtype=torch.long))
                data['seq_class_ids'].append(torch.tensor(seq_class_ids, dtype=torch.long))
                data['seq_len'].append(torch.tensor(seq_len, dtype=torch.long))
                data['transcriptions'].append(transcription)
                data['sentence_label'].append(sentence_label)
                data['session'].append(session)
                data['block_nums'].append(block_num)
                data['trial_nums'].append(trial_num)
                
                data['day_indicies'].append(torch.tensor(day, dtype=torch.long))
                data['features_len'].append(torch.tensor(neural_features.shape[0], dtype=torch.long))
                assert len(seq_class_ids) == 500

                data['seq_pad'].append(torch.ones(seq_len, dtype=torch.long))# 随便赋值为1，因为后面会被覆盖

                assert len(neural_features) > seq_len, "Feature lengths must be greater than phone sequence lengths"


        return data
    def __init__(self, data_path, ):
        self.eos_id = 42
        self.bos_id = 41
        self.pad_id = 0
        files = data_path
        files = sorted(files)
        #   'nejm-brain-to-text/data/hdf5_data_final/t15.2025.03.14/data_train.hdf5',...
    
        self.data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [],
            'seq_len': [],
            'transcriptions': [],
            'sentence_label': [],
            'session': [],
            'block_nums': [],
            'trial_nums': [],

            'day_indicies': [],
            'features_len': [],

            'seq_pad': [],
        }
        for day, file in enumerate(files):
            if os.path.exists(file):
                one_day_data = self.load_h5py_file(file, day)
                for key in self.data:
                    self.data[key].extend(one_day_data[key])
        
        target_len =2500 # 最长2144，不用complie直接3000就行
        print(f"padding for target len: {target_len}")
        for i in tqdm(range(len(self.data['neural_features']))):
            nf = self.data['neural_features'][i]
            intput_seq_len = nf.shape[0]

            if intput_seq_len < target_len:
                pad_amount = target_len - intput_seq_len
                nf_padded = np.pad(nf, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
            else:
                raise ValueError(f"Sequence length {intput_seq_len} is greater than target length {target_len}")
                # nf_padded = nf[:target_len, :]
                # assert seq_len < target_len

            # 通过索引修改原列表
            self.data['neural_features'][i] = nf_padded
            self.data['seq_pad'][i] = torch.cat([
                torch.ones(intput_seq_len, dtype=torch.long),
                torch.zeros(target_len - intput_seq_len, dtype=torch.long)
            ])

        
        #print(max(self.data['neural_features']), min(self.data['neural_features']))
        # print(f"Total data size: {len(self.data['neural_features'])}")
        # # input("data")
        # maxv = 0
        # for x in self.data['neural_features']:
        #     maxv = max(maxv, x.shape[0])
        #     print(x)
        # print(maxv)
        # input("xe")
            
    
    def __len__(self):
        return len(self.data['seq_class_ids'])
    
    def __getitem__(self, idx):
        # batch['input_features'] = pad_sequence(batch['input_features'], batch_first = True, padding_value = 0)
        # batch['seq_class_ids'] = pad_sequence(batch['seq_class_ids'], batch_first = True, padding_value = 0)

                #we need to include these keys in batch
                # features = batch['input_features'].to(self.device)
                # labels = batch['seq_class_ids'].to(self.device)
                # n_time_steps = batch['n_time_steps'].to(self.device)
                # phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                # day_indicies = batch['day_indicies'].to(self.device)
        # print(self.data['neural_features'][idx].shape)
        # print(self.data['seq_class_ids'][idx].shape)
        # print(self.data['n_time_steps'][idx])
        # print(self.data['seq_len'][idx])
        # print(self.data['day_indicies'][idx])
        # print(self.data['features_len'][idx])
        # print(self.data['seq_len'][idx])
        length = random.randint(1, self.data['seq_len'][idx])
        max_len = len(self.data['seq_class_ids'][idx])


        tgt_in = torch.cat([
            torch.tensor([self.bos_id], dtype=torch.long),
            self.data['seq_class_ids'][idx][:length],
            torch.tensor([self.pad_id] * (max_len - length - 1))
        ])
        tgt_out = torch.cat([
            self.data['seq_class_ids'][idx][:length],
            torch.tensor([self.eos_id], dtype=torch.long),
            torch.tensor([self.pad_id] * (max_len - length - 1))
        ])

        return {
            'input_features': self.data['neural_features'][idx],
            'seq_class_ids': self.data['seq_class_ids'][idx],
            'n_time_steps': self.data['n_time_steps'][idx],
            'phone_seq_lens': self.data['seq_len'][idx],
            'day_indicies': self.data['day_indicies'][idx],
            'features_len': self.data['features_len'][idx],
            
            'transcriptions': self.data['transcriptions'][idx],
            # 'sentence_label': self.data['sentence_label'][idx],
            # 'session': self.data['session'][idx],
            'block_nums': self.data['block_nums'][idx],
            'trial_nums': self.data['trial_nums'][idx],

            'tgt_in': tgt_in,
            'tgt_out': tgt_out,

            'seq_pad': self.data['seq_pad'][idx],
        }

def brain2text_test():
    from omegaconf import OmegaConf
    
    args = OmegaConf.load('rnn_args.yaml')
    val_file_paths = [os.path.join(args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in args['dataset']['sessions']]
    dataset = Brain2TextDS(data_path = val_file_paths)
    print(dataset[0]['tgt_out'])
    print(dataset[1]['tgt_out'])

if __name__ == '__main__':
    brain2text_test()

