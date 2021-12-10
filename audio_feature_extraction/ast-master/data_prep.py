# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import json
import os
import zipfile
import wget


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

    # convert the audio to 16kHz
base_dir = 'D:\\meais\\Documents\\MS\\CMU\\Courses\\11785\\Project\\AVIS_ROOT'
# os.mkdir('./data/ESC-50-master/audio_16k/')
audio_list = get_immediate_files(base_dir+'\\audio')
for audio in audio_list:
    print('ffmpeg -i ' + base_dir + '\\audio\\' + audio + ' -r 16000 ' + base_dir + '\\audio_16k\\' + audio)

    os.system("ffmpeg -i "+ base_dir + "\\audio\\" + audio + " -ar 16000 " + base_dir + '\\audio_16k\\' + audio)

    # os.system('sox ' + base_dir + '\\audio\\' + audio + ' -r 16000 ' + base_dir + '\\audio_16k\\' + audio)

# #currently using audioset labels 527
# label_set = np.loadtxt('D:\\meais\\Documents\\MS\\CMU\\Courses\\11785\\Project\\ast-master\\egs\\audioset\\data\\class_labels_indices.csv', delimiter=',', dtype='str')
# label_map = {}
# for i in range(1, len(label_set)):
#     label_map[eval(label_set[i][2])] = label_set[i][0]
# print(label_map)

# # fix bug: generate an empty directory to save json files
# if os.path.exists('./data/datafiles') == False:
#     os.mkdir('./data/datafiles')

# for fold in [1,2,3,4,5]:
#     base_path = "./data/ESC-50-master/audio_16k/"
#     meta = np.loadtxt('./data/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
#     train_wav_list = []
#     eval_wav_list = []
#     for i in range(0, len(meta)):
#         cur_label = label_map[meta[i][3]]
#         cur_path = meta[i][0]
#         cur_fold = int(meta[i][1])
#         # /m/07rwj is just a dummy prefix
#         cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
#         if cur_fold == fold:
#             eval_wav_list.append(cur_dict)
#         else:
#             train_wav_list.append(cur_dict)

#     print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

#     with open('./data/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
#         json.dump({'data': train_wav_list}, f, indent=1)

#     with open('./data/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
#         json.dump({'data': eval_wav_list}, f, indent=1)

# print('Finished ESC-50 Preparation')