# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

from src import dataloader

# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 24, 'timem': 192, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset'}

ds = dataloader.AudiosetDataset('./egs/dav/train.json', label_csv='./egs/dav/label_index.csv',
                                audio_conf=audio_conf)

train_loader = torch.utils.data.DataLoader(ds
    , batch_size=len(ds), shuffle=False, num_workers=8, pin_memory=True)
mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))