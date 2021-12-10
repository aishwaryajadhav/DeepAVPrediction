"""
YoutubeVIS data loader
"""
from pathlib import Path
import argparse
import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
import numpy as np
try:
    from start_code.transform.build_transform import build_transforms
except:
    from start_code.build_transform import build_transforms


COCO_CATEGORIES = [
    {"color": [0, 0, 0], "isthing": 1, "id": 0, "name": "background"},
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 200, 32], "isthing": 1, "id": 2, "name": "giant_panda"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "lizard"},
    {"color": [0, 100, 230], "isthing": 1, "id": 4, "name": "parrot"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "skateboard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "sedan"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "ape"},
    {"color": [0, 20, 100], "isthing": 1, "id": 8, "name": "dog"},
    {"color": [50, 0, 70], "isthing": 1, "id": 9, "name": "snake"},
    {"color": [0, 0, 192], "isthing": 1, "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "duck"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "cat"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "cow"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "fish"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "train"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "horse"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "turtle"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
    {"color": [0, 220, 176], "isthing": 1, "id": 27, "name": "surfboard"},
    {"color": [255, 99, 164], "isthing": 1, "id": 28, "name": "airplane"},
    {"color": [92, 0, 73], "isthing": 1, "id": 29, "name": "truck"},
    {"color": [133, 129, 255], "isthing": 1, "id": 30, "name": "zebra"},
    {"color": [78, 180, 255], "isthing": 1, "id": 31, "name": "tiger"},
    {"color": [0, 228, 0], "isthing": 1, "id": 32, "name": "elephant"},
    {"color": [174, 255, 243], "isthing": 1, "id": 33, "name": "snowboard"},
    {"color": [45, 89, 255], "isthing": 1, "id": 34, "name": "boat"},
    {"color": [134, 134, 103], "isthing": 1, "id": 35, "name": "shark"},
    {"color": [145, 148, 174], "isthing": 1, "id": 36, "name": "mouse"},
    {"color": [120, 166, 157], "isthing": 1, "id": 37, "name": "frog"},
    {"color": [110, 76, 0], "isthing": 1, "id": 38, "name": "eagle"},
    {"color": [174, 57, 255], "isthing": 1, "id": 39, "name": "earless_seal"},
    {"color": [110, 76, 60], "isthing": 1, "id": 40, "name": "tennis_racket"},
]

class YTVOSDataset:
    def __init__(self, root, img_folder, ann_file, transforms, return_masks, ref_frame_num, mode='train'):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.ref_frame_num = ref_frame_num
        self._transforms = transforms
        self.return_masks = return_masks
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        
        audio_dir = os.path.join(root, 'audio')
        files = os.listdir(audio_dir)
        clip_names = [file.replace('.wav', '') for file in files]
        
        self.vid_infos = []
        self.audio_files = []

        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            if info['file_names'][0][:10] in clip_names:
                self.audio_files.append(info['file_names'][0][:10]+'.wav')
                self.vid_infos.append(info)

        self.avCount = len(self.vid_infos)
        assert len(self.audio_files) == len(self.vid_infos) , "Audio Visual Lengths do not match."   
        self.img_ids = []
        
        # for all the videos that have a corr audio file, do a 75-25 videos train test split. Generate a list (img_ids) of tuples (video_index, frame_id). So each frame is a separate data point.
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))
                    
        self.audio_labels = []
        for index in range(self.avCount):
            self.audio_labels.append(list(self.getVideoLabels(index)))

        self.mode = mode

    def __len__(self):
        return len(self.img_ids) #Each frame is a separate data point


    def getVideoLabels(self, index):
        video_info = self.vid_infos[index]
        vid_id = video_info['id']

        tgt_ann = self.get_ann_info(vid_id, 0)
        tgt_labels = set(tgt_ann['labels'])
        
        for frame_id in range(1,len(video_info['filenames'])):
            tgt_ann = self.get_ann_info(vid_id, frame_id)
            if(set(tgt_labels) != set(tgt_ann['labels'])):
                print(vid_id)
                tgt_labels = tgt_labels.union(set(tgt_ann['labels']))
        
        return tgt_labels
        

    def __getitem__(self, idx):
        vid, tgt_frame_id = self.img_ids[idx]

        vid_id = self.vid_infos[vid]['id']
        vid_len = len(self.vid_infos[vid]['file_names'])
        # find adjacent frame
        ref_frame_ids = get_ref_fid(tgt_frame_id, self.ref_frame_num)
        if self.mode == 'train' or self.mode == 'val':
            tgt_ann = self.get_ann_info(vid_id, tgt_frame_id)
            tgt_labels = tgt_ann['labels']
            tgt_obj_ids = tgt_ann['obj_ids']
            tgt_is_crowd = tgt_ann['is_crowd']

            ref_labels = []
            ref_obj_ids = []
            ref_is_crowd = []
            for ref_fid in ref_frame_ids:
                ref_ann = self.get_ann_info(vid_id, ref_fid)
                ref_labels.append(ref_ann['labels'])
                ref_obj_ids.append(ref_ann['obj_ids'])
                ref_is_crowd.append(ref_ann['is_crowd'])

        else:
            # Test set
            pass
        output = dict()
        output['vid'] = torch.tensor([vid_id])
        output['tgt_fid'] = torch.tensor([tgt_frame_id])
        output['tgt_ids'] = torch.tensor(tgt_obj_ids)
        output['tgt_labels'] = tgt_labels
        for index in range(len(ref_frame_ids)):
            output[f'ref_fid{index}'] = torch.tensor([ref_frame_ids[index]])
            output[f'ref_ids{index}'] = torch.tensor(ref_obj_ids[index])
            output[f'ref_labels{index}'] = torch.tensor(ref_labels[index], dtype=torch.long)

        return output

    def get_ann_info(self, vid_id, frame_id):
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _parse_ann_info(self, ann_info, frame_id, ignore_crowd=False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_labels = []
        gt_ids = []
        # gt_masks = []
        is_crowd = []
        for i, ann in enumerate(ann_info):
            if ann['iscrowd']:
                if ignore_crowd:
                    continue
                else:
                    gt_ids.append(ann['id'])
                    gt_labels.append(self.cat2label[ann['category_id']])
                    # gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                    is_crowd.append(1)
            else:
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
                # gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                is_crowd.append(0)

        gt_labels = np.array(gt_labels, dtype=np.int64)
        ann = dict(labels=gt_labels, obj_ids=gt_ids, is_crowd=is_crowd)
        return ann

def get_ref_fid(tgt_fid, ref_num):
    if tgt_fid < ref_num:
        ref_fid = [i for i in range(tgt_fid)]
        ref_fid.extend([tgt_fid for _ in range(ref_num-tgt_fid)])
        ref_fid = ref_fid[::-1]
    else:
        ref_fid = [tgt_fid-1-i for i in range(ref_num)]
    return ref_fid


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train/JPEGImages", root / f'train2019.json'),
        "val": (root / "train/JPEGImages", root / f'train2019.json'),
        "test": (root / "train/JPEGImages", root / f'train2019.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(root, img_folder, ann_file, transforms =  None, return_masks=False,
                           ref_frame_num=args.ref_frame_num, mode=image_set)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--ytvos_path', default='D:\\meais\\Documents\\MS\\CMU\\Courses\\11785\\Project\\AVIS_ROOT', metavar='N')
    parser.add_argument('--max_det', type=int, default=10, metavar='N')
    parser.add_argument('--ref_frame_num', type=int, default=5, metavar='N')
    args = parser.parse_args()
    dataset = build('train', args)
    print(len(dataset))
    print(dataset.avCount)
    
    # count = 0
    # for ind, obj in enumerate(dataset):
    #     print(obj['tgt_labels'])

    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    #                                      drop_last=True)
    # for i, b in enumerate(loader):
    #     print(i)
    #     '''
    #     for k, v in b.items():
    #         try:
    #             print(f'{k}: {v.shape}')
    #         except:
    #             print(k)
    #     '''
    #     pass

    audio = []
    for index in range(dataset.avCount):
        wav = dataset.audio_files[index]
        labels = dataset.audio_labels[index]
        d = {"wav": wav, "labels":",".join([str(j) for j in labels])}
        audio.append(d)
    
    j = {"data":audio}

    import json

    with open('data.json', 'w') as fp:
        json.dump(j, fp)
