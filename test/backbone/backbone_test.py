'''
Program to test backbone network.
'''

import torch
import sys

## Inserting path of src directory
sys.path.insert(1, '../..')

from src.backbone import Backbone

# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
input_image = torch.randn(1,3,600,600)

backbone_obj = Backbone(model_name = 'VGG16')
out = backbone_obj.forward_pass(input_image)

## Test resnet-101
# out = backbone_obj.resnet101_backbone(input_image)
print(out.shape) 