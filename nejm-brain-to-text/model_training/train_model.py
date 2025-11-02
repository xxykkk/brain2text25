from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = OmegaConf.load('rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()