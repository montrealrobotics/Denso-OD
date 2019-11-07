"""
Faster R-CNN class
"""

import torch
import torch.nn as nn
import sys

# sys.path.insert(1, '../')
from src.backbone import Backbone
from src.rpn import RPN
from src.detection import Detector
# from src.NMS import batched_nms
import torch.nn.functional as F
from src.utils import utils

class FasterRCNN(nn.Module):
	"""docstring for generalized_faster_rcnn"""
	def __init__(self, cfg):
		super(FasterRCNN, self).__init__()
		self.cfg = cfg
		self.backbone = Backbone(self.cfg)
		# print(self.backbone)
		self.rpn = RPN(self.cfg, self.backbone.out_channels)
		# self.detector = Detector(self.cfg, subsampling_ratio) #have to find the subsampling_ratio

	def forward(self, image, gt_target, is_training):
		image_size = self.cfg.IMAGE_SIZE
		feature_map = self.backbone(image) # feature_map : [N, self.backbone_net.out_channels, H, W]
		proposals, rpn_losses = self.rpn(feature_map, gt_target, image_size, is_training) # topK proposals sorted in decreasing order of objectness score and losses: []
		final_classes, final_boxes = self.detector(feature_map, boxes)
		
		return proposals, rpn_losses

		

