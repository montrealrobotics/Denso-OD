'''
Program to test anchor generation.
'''

import torch
import sys
import numpy as np

## Inserting path of src directory
sys.path.insert(1, '../..')
from src.backbone import Backbone
from src.config import Cfg as cfg
from src.RPN import anchor_generator

# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
input_image = torch.randn(1,3,600,600)

backbone_obj = Backbone(cfg)
out = backbone_obj.forward(input_image)

anchor_generator_obj = anchor_generator()

anchors = anchor_generator_obj.get_anchors(input_image, out, cfg)

print(anchors.shape, type(anchors))
print(anchors[0,:])
## Test resnet-101
# out = backbone_obj.resnet101_backbone(input_image)
print(out.shape) 