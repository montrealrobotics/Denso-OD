'''
Contains default params for object detector
'''

import os
from yacs.config import CfgNode as CN
import argparse



conf_params = CN()

"""
For General Initialization and Settings
"""

conf_params.PATH = CN()
conf_params.PATH.DATASET = "/network/home/bansaldi/Denso-OD/datasets/kitti_dataset"
conf_params.PATH.LOGS = "/network/home/bansaldi/Denso-OD/logs"

##### Whether to use cuda or not #####
conf_params.USE_CUDA = True ## 

###### Reproducibility in randomization ######
conf_params.RANDOMIZATION = CN()
conf_params.RANDOMIZATION.SEED = 5

"""
For training
"""
conf_params.TRAIN = CN()
conf_params.TRAIN.DATASET_LENGTH = 7000
conf_params.TRAIN.BATCH_SIZE = 10
conf_params.TRAIN.EPOCHS = 50
conf_params.TRAIN.OPTIM = 'sgd' # Optimizer to use. (choices=['sgd', 'adam'])
conf_params.TRAIN.LR = 1e-3
conf_params.TRAIN.MOMENTUM = 0.9 # Used only when TRAIN.OPTIM is set to 'sgd'
conf_params.TRAIN.MILESTONES = 10,20	
conf_params.TRAIN.DSET_SHUFFLE = False
conf_params.TRAIN.FREEZE_BACKBONE = False
conf_params.TRAIN.LR_DECAY = 0.5 ## Decay learning rate by this factor every certain epochs
conf_params.TRAIN.LR_DECAY_EPOCHS = 15 	## Epochs after which we should act upon learning rate
conf_params.TRAIN.SAVE_MODEL_EPOCHS = 5 ## save model at every certain epochs
conf_params.TRAIN.DATASET_DIVIDE = 0.9 ## This fraction of dataset is for training, rest for testing.
conf_params.TRAIN.TRAIN_TYPE = 'probabilistic' ### could be ['deterministic', 'probabilistic']

"""
For Testing
"""
conf_params.TEST = CN()
conf_params.TEST.DETECTIONS_PER_IMAGE = 50

"""
For Backbone
"""

conf_params.BACKBONE = CN()
### choices = ['VGG16', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
conf_params.BACKBONE.MODEL_NAME = 'resnet50'
### choices = [1,2,3,4]
conf_params.BACKBONE.RESNET_STOP_LAYER = 3


"""
For Data 
"""
conf_params.INPUT = CN()
conf_params.INPUT.IMAGE_SIZE = (375, 1242)
# Params to define input image transformation.
# The input to backbone network has to be RGB 
# image with intensity scaled between 0 to 1.
# The following normalization is applied on the 
# top of that image. 

# The mean/std are standard used for training the 
# backbone networks. 

# Find more details here.
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
# https://pytorch.org/docs/stable/torchvision/models.html

# The mean and std are of ImageNet dataset, as the models are trained on that. 
# We will stick to same to get better output.
conf_params.INPUT.MEAN = [0.485, 0.456, 0.406]
conf_params.INPUT.STD = [0.229, 0.224, 0.225]
conf_params.INPUT.LABELS_TO_TRAIN = ['Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Person_sitting', 'Cyclist']
conf_params.INPUT.NUM_CLASSES = 7


"""
For Anchor Generator7
"""
conf_params.ANCHORS = CN()
conf_params.ANCHORS.ASPECT_RATIOS = [0.5,1,2]
conf_params.ANCHORS.ANCHOR_SCALES = [64, 128, 256]
conf_params.ANCHORS.POS_PROPOSAL_THRES = 0.7
conf_params.ANCHORS.NEG_PROPOSAL_THRES = 0.3


"""
For Region Proposal Network
"""
conf_params.RPN = CN()
conf_params.RPN.N_ANCHORS_PER_LOCATION = 9
conf_params.RPN.CONV_MEAN = 0.01
conf_params.RPN.CONV_VAR = 0.02
conf_params.RPN.BIAS = 0.01
conf_params.RPN.UNCERTAIN_MEAN = 0.01
conf_params.RPN.UNCERTAIN_VAR = 0.02
conf_params.RPN.UNCERTAIN_BIAS = 0.01
conf_params.RPN.ACTIVATION_ALPHA = 1
conf_params.RPN.LOSS_WEIGHT = 1.0
conf_params.RPN.BATCH_SIZE_PER_IMAGE = 256
conf_params.RPN.NMS_THRESH = 0.7
conf_params.RPN.POSITIVE_FRACTION = 0.5
conf_params.RPN.MIN_SIZE_PROPOSAL = 5
conf_params.RPN.PRE_NMS_TOPK_TRAIN = 12000
conf_params.RPN.PRE_NMS_TOPK_TEST = 6000
conf_params.RPN.POST_NMS_TOPK_TRAIN = 2000
conf_params.RPN.POST_NMS_TOPK_TEST = 1000
conf_params.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
conf_params.RPN.IOU_THRESHOLDS = [0.3, 0.7]
conf_params.RPN.IOU_LABELS = [0, -1, 1]
conf_params.RPN.BOUNDARY_THRESH = -1
conf_params.RPN.SMOOTH_L1_BETA = 0.0


"""
For ROI and Detection
"""
conf_params.ROI_HEADS = CN()
conf_params.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
conf_params.ROI_HEADS.POSITIVE_FRACTION = 0.25
conf_params.ROI_HEADS.SCORE_THRESH_TEST = 0.5
conf_params.ROI_HEADS.NMS_THRESH_TEST = 0.5
conf_params.ROI_HEADS.PROPOSAL_APPEND_GT = True
conf_params.ROI_HEADS.IOU_THRESHOLDS = [0.5]
conf_params.ROI_HEADS.IOU_LABELS = [0, 1]
# conf_params.ROI_HEADS.POOLER_TYPE = "ROIPool"
conf_params.ROI_HEADS.POOLER_TYPE = "ROIAlign"
conf_params.ROI_HEADS.FC_DIM = 1024
conf_params.ROI_HEADS.CLS_AGNOSTIC_BBOX_REG = True
conf_params.ROI_HEADS.SMOOTH_L1_BETA = 0.0
conf_params.ROI_HEADS.POOLER_RESOLUTION = 14 # After this there is MaxPool2D, so final resolution is 7x7
conf_params.ROI_HEADS.POOLER_SAMPLING_RATIO = 0
conf_params.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 10.0, 10.0)
# conf_params.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

