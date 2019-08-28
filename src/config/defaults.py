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






