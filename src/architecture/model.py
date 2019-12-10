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
		self.detector = Detector(self.cfg, self.backbone.stride, self.backbone.out_channels)


	def forward(self, image, gt_target, is_training):
		"""
		Args:
			image: 	   Tensor[N,H,W,C], N is batch_size
			gt_target: Tensor[Instances], each instance has attribute 'gt_boxes' and 'gt_classes'
		
		Returns:
			rpn_proposals:  List[Instances], length of list is N. Each instance of the list have attribute 'proposal_boxes' and 'objectness_logits'
			rpn_losses: 	Dict, the dict have two losses - classification_loss and regression loss  
			prediction:		List[Instances], length of list is N. Each instance stores the topk most confidence detections. 
							It has attributes - 'pred_boxes', 'scores' , 'pred_sigma', 'pred_classes'
			detection_loss: Dict, classfication loss and bounding box loss

		"""

		image_size = image.shape[-2:]
		
		feature_map = self.backbone(image) # feature_map : [N, self.backbone_net.out_channels, H, W]

		rpn_proposals, rpn_losses = self.rpn(feature_map, gt_target, image_size, is_training) # topK proposals sorted in decreasing order of objectness score and losses: []
		
		detections, detection_loss = self.detector(feature_map, rpn_proposals, gt_target, is_training)

		return rpn_proposals, detections, rpn_losses, detection_loss 

		

