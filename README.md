# DeepAVPrediction

## Image Segmentation

To train the baseline Mask-RCNN model, run the following from the **image_segmentation/mask-rcnn-images** directory:

```
python3 coco.py train --dataset={PATH/TO/DATA/DIRECTORY} --model=coco --logs=image_model
```

To evaluate the model:

```
python3 coco.py evaluate --dataset={PATH/TO/DATA/DIRECTORY} --model=last --logs=image_model
```

To train the image + audio model, run the following from the **image_segmentation/mask-rcnn-audio** directory:

```
python3 coco.py train --dataset={PATH/TO/DATA/DIRECTORY} --model=coco --logs=audio_model
```

To evaluate the model:

```
python3 coco.py evaluate --dataset={PATH/TO/DATA/DIRECTORY} --model=last --logs=audio_model
```

## Video Instance Segmentation 

In order to setup, train and evaluate the model we use the same framework as CrossVIS since our code is built on top of it. Please refer to CrossVis project link for the same - https://github.com/hustvl/CrossVIS. Only additional thing our project requires is to have train and validation audio pickle file inside video_segmentation folder. 

## References
https://github.com/matterport/Mask_RCNN
