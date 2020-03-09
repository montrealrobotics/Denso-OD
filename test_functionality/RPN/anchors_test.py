'''
Program to test anchor generation.
'''

import torch
import sys
import numpy as np
import matplotlib.image as mpimg ## To load the image

## Inserting path of src directory
sys.path.insert(1, '../..')
from src.backbone import Backbone
from src.config import Cfg as cfg
from src.RPN import anchor_generator
from src.preprocess import image_transform ## It's a function, not a class. 


# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)


img = mpimg.imread('../preprocess/test.jpg')		## Gives RGB image of dimension H x W x C with inten values between 0-255

transform = image_transform(cfg)
input_image = transform(img)
input_image = torch.unsqueeze(input_image, dim=0)

backbone_obj = Backbone(cfg)
out = backbone_obj.forward(input_image)

anchor_generator_obj = anchor_generator()

anchors = anchor_generator_obj.get_anchors(input_image, out, cfg)

print(anchors.shape, type(anchors))
print(anchors[0,:])
## Test resnet-101
# out = backbone_obj.resnet101_backbone(input_image)
print(out.shape) 