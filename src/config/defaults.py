'''
Contains default params for object detector
'''

import os
from yacs.config import CfgNode as CN


## Intiailizing..
conf_params = CN()

##### BACKBONE CONFIG #####
"""
Contains all the params that defines 
our backbone network!
"""

conf_params.BACKBONE = CN()

### choices = ['VGG16', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
conf_params.BACKBONE.MODEL_NAME = 'resnet101'

### choices = [1,2,3,4]
conf_params.BACKBONE.RESNET_STOP_LAYER = 4 


##### Input config #####
"""
Params to define input image transformation.
The input to backbone network has to be RGB 
image with intensity scaled between 0 to 1.
The following normalization is applied on the 
top of that image. 

The mean/std are standard used for training the 
backbone networks. 

Find more details here.
https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
https://pytorch.org/docs/stable/torchvision/models.html

The mean and std are of ImageNet dataset, as the models are trained on that. 
We will stick to same to get better output.
"""

conf_params.INPUT = CN()
conf_params.INPUT.MEAN = 0.485, 0.456, 0.406
conf_params.INPUT.STD = 0.229, 0.224, 0.225


##### ANCHOR CONFIG #####
"""
Necessary params to define anchors
"""
conf_params.ANCHORS = CN()
conf_params.ANCHORS.ASPECT_RATIOS = 0.5, 1, 2
conf_params.ANCHORS.ANCHOR_SCALES = 8, 16, 32
conf_params.ANCHORS.N_ANCHORS_PER_LOCATION = 9


##### REGION PROPOSAL NETWORK CONFIG #####
"""
Used for defining region proposal network
"""
conf_params.RPN = CN()
conf_params.RPN.OUT_CHANNELS = 512
conf_params.RPN.N_ANCHORS_PER_LOCATION = 9






