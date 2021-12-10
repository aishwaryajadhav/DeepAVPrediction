# python3 coco.py train --dataset=../train --model=coco --year=2021 --logs=audio_pretrained > audio_pretrained_v1.txt
# python3 coco.py evaluate --dataset=../train --model=last --year=2021 --logs=audio_pretrained --limit=100 > audio_eval_v1.txt
# python3 coco.py train --dataset=../train --model=last --year=2021 --logs=audio_pretrained > audio_pretrained_v2.txt
"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import re
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import pickle

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import skimage.color
import skimage.io
import skimage.transform
# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # COCO has 80 classes


############################################################
#  Dataset
############################################################
import pickle

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        audio_data = pickle.load(open("../audio_layerwise_features.pickle", 'rb'))
        audio_dict = dict(zip(audio_data['audio_name'], audio_data['1d_embedding']))
        self.audios = audio_dict
        
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/JPEGImages".format(dataset_dir)
        
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())
        # print("CLASS_IDS:",len(class_ids))
            
        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())
        image_ids = list(coco.imgs.keys()) # CHANGED
        
        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            # print('path: ', image_dir, coco.imgs[i]['file_name'])
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.                                                                                                                                       
        """
        # Load image
        image_path = self.image_info[image_id]['path']

        audio = self.get_audio(image_path)
        
        # skip images with no audio
        if audio is None:
            return None, None
        # print('AUDIO_SHAPE: ', audio.shape)
        
        image = skimage.io.imread(image_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # print('IMAGE_SHAPE: ', image.shape)
        return image, audio[None, ::]

    def get_audio_path(self, image_path):
        audio_key = re.sub('\.\./train/JPEGImages/', '', image_path)
        audio_key = re.sub('/[0-9]+\.jpg', '', audio_key) + '.wav'
        return audio_key

    def get_audio(self, image_path):
        audio_key = self.get_audio_path(image_path)
        if audio_key in self.audios:
            return self.audios[audio_key]
        else:
            return None
    
    def audio_exists(self, image_path):
        audio_key = self.get_audio_path(image_path)
        if audio_key in self.audios:
            return True
        return False
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="segm", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image, audio = dataset.load_image(image_id)
        if image is None:
            print("EVAL NONE")
            continue
        # Run detection
        t = time.time()
        r = model.detect([image], [audio], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    f = open("temp.pkl", 'wb')
    pickle.dump(results, f)
    f.close()
    
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model
    
    # Load weights
    # print("Loading weights ", model_path)
    # exclude_layers = ['input_gt_boxes', 'input_gt_class_ids', 'input_gt_masks', 'input_image', 'input_image_meta', 'input_rpn_bbox', 'input_rpn_match', 'lambda_1', 'lambda_4', 'max_pooling2d_1', 'mrcnn_bbox', 'mrcnn_bbox_fc', 'mrcnn_bbox_loss', 'mrcnn_class', 'mrcnn_class_bn1', 'mrcnn_class_bn2', 'mrcnn_class_conv1', 'mrcnn_class_conv2', 'mrcnn_class_logits', 'mrcnn_class_loss', 'mrcnn_mask', 'mrcnn_mask_bn1', 'mrcnn_mask_bn2', 'mrcnn_mask_bn3', 'mrcnn_mask_bn4', 'mrcnn_mask_conv1', 'mrcnn_mask_conv2', 'mrcnn_mask_conv3', 'mrcnn_mask_conv4', 'mrcnn_mask_deconv', 'mrcnn_mask_loss', 'output_rois', 'pool_squeeze', 'proposal_targets', 'res2a_branch1', 'res2a_branch2a', 'res2a_branch2b', 'res2a_branch2c', 'res2a_out', 'res2b_branch2a', 'res2b_branch2b', 'res2b_branch2c', 'res2b_out', 'res2c_branch2a', 'res2c_branch2b', 'res2c_branch2c', 'res2c_out', 'res3a_branch1', 'res3a_branch2a', 'res3a_branch2b', 'res3a_branch2c', 'res3a_out', 'res3b_branch2a', 'res3b_branch2b', 'res3b_branch2c', 'res3b_out', 'res3c_branch2a', 'res3c_branch2b', 'res3c_branch2c', 'res3c_out', 'res3d_branch2a', 'res3d_branch2b', 'res3d_branch2c', 'res3d_out', 'res4a_branch1', 'res4a_branch2a', 'res4a_branch2b', 'res4a_branch2c', 'res4a_out', 'res4b_branch2a', 'res4b_branch2b', 'res4b_branch2c', 'res4b_out', 'res4c_branch2a', 'res4c_branch2b', 'res4c_branch2c', 'res4c_out', 'res4d_branch2a', 'res4d_branch2b', 'res4d_branch2c', 'res4d_out', 'res4e_branch2a', 'res4e_branch2b', 'res4e_branch2c', 'res4e_out', 'res4f_branch2a', 'res4f_branch2b', 'res4f_branch2c', 'res4f_out', 'res4g_branch2a', 'res4g_branch2b', 'res4g_branch2c', 'res4g_out', 'res4h_branch2a', 'res4h_branch2b', 'res4h_branch2c', 'res4h_out', 'res4i_branch2a', 'res4i_branch2b', 'res4i_branch2c', 'res4i_out', 'res4j_branch2a', 'res4j_branch2b', 'res4j_branch2c', 'res4j_out', 'res4k_branch2a', 'res4k_branch2b', 'res4k_branch2c', 'res4k_out', 'res4l_branch2a', 'res4l_branch2b', 'res4l_branch2c', 'res4l_out', 'res4m_branch2a', 'res4m_branch2b', 'res4m_branch2c', 'res4m_out', 'res4n_branch2a', 'res4n_branch2b', 'res4n_branch2c', 'res4n_out', 'res4o_branch2a', 'res4o_branch2b', 'res4o_branch2c', 'res4o_out', 'res4p_branch2a', 'res4p_branch2b', 'res4p_branch2c', 'res4p_out', 'res4q_branch2a', 'res4q_branch2b', 'res4q_branch2c', 'res4q_out', 'res4r_branch2a', 'res4r_branch2b', 'res4r_branch2c', 'res4r_out', 'res4s_branch2a', 'res4s_branch2b', 'res4s_branch2c', 'res4s_out', 'res4t_branch2a', 'res4t_branch2b', 'res4t_branch2c', 'res4t_out', 'res4u_branch2a', 'res4u_branch2b', 'res4u_branch2c', 'res4u_out', 'res4v_branch2a', 'res4v_branch2b', 'res4v_branch2c', 'res4v_out', 'res4w_branch2a', 'res4w_branch2b', 'res4w_branch2c', 'res4w_out', 'res5a_branch1', 'res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c', 'res5a_out', 'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c', 'res5b_out', 'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c', 'res5c_out', 'roi_align_classifier', 'roi_align_mask', 'rpn_bbox', 'rpn_bbox_loss', 'rpn_class', 'rpn_class_logits', 'rpn_class_loss', 'rpn_model', 'zero_padding2d_1']
    # model.load_weights(model_path, by_name=True) #, exclude=exclude_layers) #CHANGED
    # exclude_layers = ['mrcnn_bbox_fc', 'mrcnn_mask', 'mrcnn_class_logits']
    model.load_weights(model_path, by_name=True) #, exclude=exclude_layers) #CHANGED 
    
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***
        '''
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10, #40, # CHANGED
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30, #120
                    layers='4+',
                    augmentation=augmentation)
        '''
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))