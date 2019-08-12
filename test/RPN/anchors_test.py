'''
Program to test backbone network.
'''

import torch
import sys
import numpy as np

## Inserting path of src directory
sys.path.insert(1, '../..')
from src.backbone import Backbone
from src.RPN import anchor_generator

# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
input_image = torch.randn(1,3,600,600)

backbone_obj = Backbone(model_name = 'resnet101')
out = backbone_obj.forward_pass(input_image)

anchor_generator_obj = anchor_generator()

anchors = anchor_generator_obj.get_anchors(input_image, out, aspect_ratios = [0.5, 1, 2, 4, 8])

print(anchors.shape, type(anchors))

## Test resnet-101
# out = backbone_obj.resnet101_backbone(input_image)
print(out.shape) 