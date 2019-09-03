'''
Program to test RPN target generation.
'''

import torch
import sys
import numpy as np
import matplotlib.image as mpimg ## To load the image

## Inserting path of src directory
sys.path.insert(1, '../..')
from src.backbone import Backbone
from src.config import Cfg as cfg
from src.RPN import *
from src.preprocess import preprocess_image ## It's a function, not a class. 
from src.loss import RPNLoss
# from src.RPN import RPN
torch.manual_seed(1)
np.random.seed(1)
torch.set_default_tensor_type('torch.FloatTensor') 
torch.set_default_dtype(torch.float32)


# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
img = mpimg.imread('../preprocess/test.jpg')		## Gives RGB image of dimension H x W x C with inten values between 0-255
input_image = preprocess_image(cfg, img)
print(input_image.shape)
bbox = np.array([[20,30,400,500], [300,400,500,600], [100,200,500,600], [400,400,500,500]]) ## y1, x1, y2, x2 format!
labels = np.array([2,7,3,2])
targets = {'boxes':bbox, 'labels':labels}

backbone_obj = Backbone(cfg)
rpn_model = RPN(backbone_obj.out_channels, cfg)
loss_object = RPNLoss()
out = backbone_obj.forward(input_image)
prediction = rpn_model.forward(out)


print("shape of rpn_output is: ", prediction['bbox_pred'].size(), prediction['bbox_class'].size())

rpn_target = RPN_targets(cfg)
valid_anchors, valid_labels = rpn_target.get_targets(input_image, out, targets)
target = {}
target['gt_bbox'] = torch.unsqueeze(torch.from_numpy(valid_anchors),0)
target['gt_anchor_label'] = torch.unsqueeze(torch.from_numpy(valid_labels).long(), 0) 
valid_indices = np.where(valid_labels != -1)

print("Shape of target output is: ", target['gt_bbox'].shape, target['gt_anchor_label'].shape)

## Test resnet-101
print(out.shape) 

prediction['bbox_pred'] = prediction['bbox_pred'].type('torch.FloatTensor')
prediction['bbox_uncertainty_pred'] = prediction['bbox_uncertainty_pred'].type('torch.FloatTensor')
prediction['bbox_class'] = prediction['bbox_class'].type('torch.FloatTensor')
target['gt_bbox'] = target['gt_bbox'].type('torch.FloatTensor')
target['gt_anchor_label'] = target['gt_anchor_label'].type('torch.LongTensor')

## Test loss
loss = loss_object.compute_loss(prediction, target, valid_indices)
print(loss.item(), loss, loss.type())
print(loss_object.pos_anchors, loss_object.neg_anchors)
loss.backward()
print('loss backward successful!')