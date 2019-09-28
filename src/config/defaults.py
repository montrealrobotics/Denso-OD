'''
Contains default params for object detector
'''

import os
from yacs.config import CfgNode as CN


## Intiailizing..
conf_params = CN()

##### Whether to use cuda or not #####
conf_params.USE_CUDA = False ## False by default, to be changed to True in the code if cuda is available

#### IF True, we won't use GPU even if it would be available
#### IF False, we will use GPU only if available	
conf_params.NO_GPU = False	


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

##### Datatypes #####
conf_params.DTYPE = CN()
conf_params.DTYPE.FLOAT = "torch.FloatTensor"
conf_params.DTYPE.LONG = "torch.LongTensor"

##### ANCHOR CONFIG #####
"""
Necessary params to define anchors
"""
conf_params.ANCHORS = CN()
conf_params.ANCHORS.ASPECT_RATIOS = 0.5, 1, 2
conf_params.ANCHORS.ANCHOR_SCALES = 128, 256, 512
conf_params.ANCHORS.N_ANCHORS_PER_LOCATION = 9


##### REGION PROPOSAL NETWORK CONFIG #####
"""
Used for defining region proposal network
"""
conf_params.RPN = CN()
conf_params.RPN.OUT_CHANNELS = 512
conf_params.RPN.N_ANCHORS_PER_LOCATION = 9
conf_params.RPN.SOFTPLUS_BETA = 1
conf_params.RPN.SOFTPLUS_THRESH = 20

"""
To be used for initilizing RPN weights
"""
conf_params.RPN.CONV_MEAN = 0
conf_params.RPN.CONV_VAR = 0.01
conf_params.RPN.BIAS = 0
conf_params.RPN.UNCERTAIN_MEAN = 0
conf_params.RPN.UNCERTAIN_VAR = 0.01
conf_params.RPN.UNCERTAIN_BIAS = 300 ## Keeping it high to avoid running into NaN losses
conf_params.RPN.ACTIVATION_ALPHA = 1



"""
For training
"""
conf_params.TRAIN = CN()
conf_params.TRAIN.ADAM_LR = 1e-4
conf_params.TRAIN.EPOCHS = 60
conf_params.TRAIN.DSET_SHUFFLE = True
conf_params.TRAIN.BATCH_SIZE = 1 ## Because all the images are of different sizes. 
conf_params.TRAIN.FREEZE_BACKBONE = False
conf_params.TRAIN.LR_DECAY = 0.1 ## Decay learning rate by this factor every certain epochs
conf_params.TRAIN.LR_DECAY_EPOCHS = 15 	## Epochs after which we should act upon learning rate
conf_params.TRAIN.TRAIN_TYPE = 'deterministic' ### could be ['deterministic', 'probabilistic']
conf_params.TRAIN.DATASET_DIVIDE = 0.7 ## This fraction of dataset is for training, rest for testing.
conf_params.TRAIN.NUSCENES_IMAGE_RESIZE_FACTOR = 1.5 ## The image size will be reduced for Nuscenes dataset by this amount