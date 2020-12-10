# TinyVOCSegmentation

This is a task of instance segmentation with Tiny PASCAL VOC dataset. There are 20 object classes in the dataset. We use 1,200 images for training, 149 for validation, and 100 for testing. Use Mask RCNN model from GitHub [1] and train with the weights pre-trained on ImageNet. The highest testing mAP50 can reach 27.19%.

# Reproducing Submission
1.  [Requirement](#Requirement)
2. [Installation](#Installation)
3. [Dataset Configuration](#Dataset-Configuration)
4. [Training](#Training)
5. [Inference](#Inference)

# Requirement
This [Mask RCNN](https://github.com/matterport/Mask_RCNN) model use tensorflow for training. Since it could not support tensorflow 2.X, there are some environment requirement.
* Python 3.5
* cuda 10.0
* tensorflow 1.15.0

# Installation
1. Clone this repository. 
```
git clone https://github.com/qaz812345/TinyVOCSegmentation.git
```
2. cd to this repository and install the pytyhon packages with requirements.txt.
```
cd TinyVOCSegmentation
pip install -r requirements.txt
```
3. Clone [Mask RCNN](https://github.com/matterport/Mask_RCNN) repository.
```
git clone https://github.com/matterport/Mask_RCNN.git
```
4. Move ```tinyvoc``` directory to ```Mask_RCNN/samples/```.
5. Create log directory.
```mkdir logs```

# Dataset Configuration
1. Download ```train_images.zip```, ```test_images.zip```, ```pascal_train.json```,and ```test.json``` from [Google drive](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK).
2. Unzip the dataset files.
3. Order the training images by file name and pick the last 149 files as the vaildation set. Move these images to ```val``` directory.
4. Set the data directory structure as:
```
TinyVOCSegmentation
|__Mask_RCNN
|   |__images
|   |   |__voc
|   |   |   |__pascal_train.json
|   |   |   |__test.json 
|   |   |   |__train
|   |   |   |   |__<test_image>
|   |   |   |   |__ ...
|   |   |   |
|   |   |   |__val
|   |   |   |   |__<val_image>
|   |   |   |   |__ ...
|   |   |   |
|   |   |   |__test
|   |   |      |__<test_image>
|   |   |      |__ ...
|   |   |  
|   |   |__ ...
|   |   |__ ...
```

# Training
### Model Configuration
Use ResNet50 as the backbone, and train with weights pre-trained on ImageNet.

### Data Pre-process and Augmentation
*	Resize image and pad with zeros to square
* Random horizontal flip (p = 0.5)
*	Random crops (percent = (0, 0.1))
*	Random Gaussian blur (sigma = (0, 0.5))
*	Contrast normalization
*	Random color change
*	Random affine transformations (scale, translate, rotate, shear)

### Hyperparameters
*	Epochs = 80
* Stage 1: Train heads for 40 epochs
* Stage 2: Fine tune layers from ResNet stage 4 and up for 20 epochs
* Stage 3: Fine tune all layers for 20 epochs

*	Batch size = 4
*	Optimizer = SGD (learning rate=0.001, momentum=0.9, weight decay=0.0001)
*	Loss function = rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss
* GPU counts = 2
*	Image min dim = 256
*	Image max dim = 512
*	Detection min confidence = 0.9

You can take a look at ```Mask_RCNN/mrcnn/config.py``` for other hyperparameters.

### Train
Training command:
```
python samples/tinyvoc/tinyvoc.py train --dataset=images/voc/ --weights=imagenet
```

# Inference
Output structure of json file:
```
[{"image_id": id of test image, which is the key in "test.json", int
  ,"score": probability for the class of this instance, float
  ,"category_id": category id of this instance, int
  ,"segmentation": encode the mask in RLE by provide function, str} # a dict for one instance
  , ...
 ]
```
Inference command:
```
python samples/tinyvoc/tinyvoc.py test --dataset images/voc --weights logs/<experiment_id>/<checkpoint_file_name> --save <result_save_path>
```

# Reference
*	matterport, Mask_RCNN, viewed 10 Dec 2020, [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
*	matterport, issues#281, viewed 10 Dec 2020, [https://github.com/matterport/Mask_RCNN/issues/281](https://github.com/matterport/Mask_RCNN/issues/281)
*	genausz, voc.py, viewed 10 Dec 2020, [https://github.com/genausz/Mask_RCNN/blob/master/samples/voc/voc.py](https://github.com/genausz/Mask_RCNN/blob/master/samples/voc/voc.py)

