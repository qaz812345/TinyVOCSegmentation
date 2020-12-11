"""
Mask R-CNN
Train on the Tiny PASCAL VOC dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modify by Yu Jou Chen

------------------------------------------------------------

Usage: import the module, or run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 tinyvoc.py train --dataset=/path/to/tinyvoc/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 tinyvoc.py train --dataset=/path/to/tinyvoc/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 tinyvoc.py train --dataset=/path/to/tinyvoc/dataset --weights=imagenet

    # Apply color splash to an image
    python3 tinyvoc.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Test using the last weights you trained
    python3 tinyvoc.py test --dataset=/path/to/tinyvoc/test/dataset --weights=last
"""

import os
import cv2
import sys
import json
import datetime
import numpy as np
import skimage.draw
from itertools import groupby
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import imgaug.augmenters as iaa
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"


############################################################
#  Configurations
############################################################

class VOCConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "voc"

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 2

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # Background + 20 object classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 3000


############################################################
#  Dataset
############################################################

class VOCDataset(utils.Dataset):
    def load_voc(self, dataset_dir, subset):
        """Load a subset of the VOC dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Load label file
        self.voc = COCO(os.path.join(dataset_dir, "pascal_train.json"))

        # Add classes.
        for i in range(1, len(self.voc.cats)+1):
            self.add_class('voc', self.voc.cats[i]['id'], self.voc.cats[i]['name'])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # PASCAL VOC dataset saves each image in the form:
        # { 736: {'file_name': '2009_001816.jpg',
        #           'id': 736,
        #           'height': 375,
        #           'width': 500
        #           },
        #   ... {
        #    
        #       }
        # }

        # Get all keys of images
        annos = list(self.voc.imgs.values())
        annos = sorted(annos, key=lambda k: k['file_name'])
        if subset == "train":
            annos= annos[:1200]
        if subset == "val":
            annos = annos[1200:]
        
        # Add images
        for anno in annos:
            # load_mask() needs the image size to convert polygons to masks.
            image_path = os.path.join(dataset_dir, anno['file_name'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                'voc',
                image_id=anno['file_name'],
                id=anno['id'],
                path=image_path,
                width=anno['width'],
                height=anno['height'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Use the imgIds to find all instance ids of the image
        img_info = self.image_info[image_id]
        ann_ids = self.voc.getAnnIds(imgIds=img_info['id'])
        anns = self.voc.loadAnns(ann_ids)
        
        # [height, width, instance_count]
        masks = np.zeros([img_info["height"], img_info["width"], len(ann_ids)], dtype=np.uint8)
        classes = []

        for i in range(len(ann_ids)):
            masks[:, :, i] = self.voc.annToMask(anns[i])
            classes.append(anns[i]['category_id'])

        # Return mask, and array of class IDs of each instance.
        return masks.astype(np.bool), np.array(classes, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        img_info = self.image_info[image_id]
        if img_info["source"] == "voc":
            return img_info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.
        Return: binary mask: numpy 2D array
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
        """Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        Return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = VOCDataset()
    dataset_train.load_voc(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VOCDataset()
    dataset_val.load_voc(args.dataset, "val")
    dataset_val.prepare()

    # Image Augmentation
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            shear=(-2, 2))
        ], random_order=True) # apply augmenters in random order

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=60,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=100,
                layers='all',
                augmentation=augmentation)


############################################################
#  VOC Test
############################################################

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskUtils.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "%s/splash_{:%Y%m%dT%H%M%S}.png".format(save_dir, datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def detect(model, dataset_dir, save_dir='logs/dection'):
    # Run model detection
    # Load test dataset
    voc_test = COCO(os.path.join(dataset_dir, 'test.json'))
    voc_dt = []
    # Read image
    for imgid in vocGt.imgs:
        image = cv2.imread(os.path.join(dataset_dir, "test", voc_test.loadImgs(ids=imgid)[0]['file_name']))[:,:,::-1] # load image
        # Detect objects
        result = model.detect([image], verbose=1)[0]
        masks, categories, scores = result['masks'], result['class_ids'], result['scores']
        n_instances = len(scores)
        print(n_instances)
        for i in range(n_instances): # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid # this imgid must be same as the key of test.json
            pred['score'] = float(scores[i])
            pred['category_id'] = int(categories[i])
            pred['segmentation'] = binary_mask_to_rle(masks.astype(np.uint8)[:,:,i]) # save binary mask to RLE, e.g. 512x512 -> rle
            voc_dt.append(pred)

        # Save output images
        splash = color_splash(image, result['masks'])
        file_name = voc_test.loadImgs(ids=imgid)[0]['file_name']
        save_path = "{}/splash_{}.png".format(save_dir, file_name)
        skimage.io.imsave(save_path, splash)
    
    with open("{}/submission.json".format(save_dir), "w") as f:
        json.dump(voc_dt, f)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'test' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/voc/dataset/",
                        help='Directory of the VOC dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--save', required=False,
                        metavar="path for saving detected image",
                        help='Path for saving detected image')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VOCConfig()
    else:
        class InferenceConfig(VOCConfig):
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
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        detect(model, args.dataset, args.save)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
