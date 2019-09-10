'''
Program to test backbone network.
'''

import torch
import sys

## Inserting path of src directory
sys.path.insert(1, '../..')
from src.config import Cfg as cfg
from src.backbone import Backbone

# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
input_image = torch.randn(1,3,800,800)
cfg.BACKBONE.RESNET_STOP_LAYER = 1
cfg.BACKBONE.MODEL_NAME = 'resnet152'
backbone_obj = Backbone(cfg)
print(backbone_obj.out_channels)
out = backbone_obj(input_image)

## Test resnet-101
# out = backbone_obj.resnet101_backbone(input_image)
print(out.shape) 