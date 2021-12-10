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


## References
https://github.com/matterport/Mask_RCNN
