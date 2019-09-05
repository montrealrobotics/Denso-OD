"""
Training RPNs. 
"""

import torch
import sys
import numpy as np
import matplotlib.image as mpimg ## To load the image

## Inserting path of src directory
sys.path.insert(1, '../')
from src.architecture import FRCNN
from src.config import Cfg as cfg
from src.RPN import anchor_generator, RPN_targets
from src.preprocess import image_transform ## It's a function, not a class. 
from src.loss import RPNLoss
from torchvision import datasets as dset

# Setting the seeds
torch.manual_seed(1)
np.random.seed(1)

## setting default variable types
torch.set_default_tensor_type('torch.FloatTensor') 
torch.set_default_dtype(torch.float32)

### use cuda only if it's available and permitted
if torch.cuda.is_available() and not cfg.NO_GPU:
	cfg.USE_CUDA = True

### let's generate the dataset
tranform = image_transform(cfg)
coco_dataset = coco = dset.CocoDetection('/home/coco_dataset_new/train2017_modified/', '/home/coco_dataset_new/instances_train2017_modified.json', transform= tranform) 
trainloader = torch.utils.data.DataLoader(coco_dataset, batch_size=1, shuffle=True)

# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
image, labels = iter(trainloader).next()
print(image)
target = process_labels(labels)

# print(img.shape)
# bbox = np.array([[20,30,400,500], [300,400,500,600], [100,200,500,600], [400,400,500,500]]) ## y1, x1, y2, x2 format!
# labels = np.array([2,7,3,2])	
# targets = {'boxes':bbox, 'labels':labels}

# backbone_obj = Backbone(cfg)
# rpn_model = RPN(backbone_obj.out_channels, cfg)
# loss_object = RPNLoss(cfg)

# if cfg.USE_CUDA:
# 	backbone_obj = backbone_obj.cuda()
# 	rpn_model = rpn_model.cuda()
# 	loss_object = loss_object.cuda()
# 	input_image = input_image.cuda()

# out = backbone_obj.forward(input_image)	
# prediction = rpn_model.forward(out)

# if cfg.USE_CUDA:
# 	cfg.DTYPE.FLOAT = 'torch.cuda.FloatTensor'
# 	cfg.DTYPE.LONG = 'torch.cuda.LongTensor'

# # dtypefloat = 'torch.FloatTensor'
# # dtypelong = 'torch.LongTensor'

# print("shape of rpn_output is: ", prediction['bbox_pred'].size(), prediction['bbox_class'].size())

# rpn_target = RPN_targets(cfg)
# valid_anchors, valid_labels = rpn_target.get_targets(input_image, out, targets)
# target = {}
# target['gt_bbox'] = torch.unsqueeze(torch.from_numpy(valid_anchors),0)
# target['gt_anchor_label'] = torch.unsqueeze(torch.from_numpy(valid_labels).long(), 0) 
# valid_indices = np.where(valid_labels != -1)

# print("Shape of target output is: ", target['gt_bbox'].shape, target['gt_anchor_label'].shape)

# ## Test resnet-101
# print(out.shape) 

# prediction['bbox_pred'] = prediction['bbox_pred'].type(cfg.DTYPE.FLOAT)
# prediction['bbox_uncertainty_pred'] = prediction['bbox_uncertainty_pred'].type(cfg.DTYPE.FLOAT)
# prediction['bbox_class'] = prediction['bbox_class'].type(cfg.DTYPE.FLOAT)
# target['gt_bbox'] = target['gt_bbox'].type(cfg.DTYPE.FLOAT)
# target['gt_anchor_label'] = target['gt_anchor_label'].type(cfg.DTYPE.LONG)

# ## Test loss
# loss = loss_object(prediction, target, valid_indices)
# print(loss.item(), loss, loss.type())
# print(loss_object.pos_anchors, loss_object.neg_anchors)
# loss.backward()
# print('loss backward successful!')