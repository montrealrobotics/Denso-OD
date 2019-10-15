"""
Generalized faster R-CNN class
"""


import torch
import torch.nn as nn
import sys
sys.path.insert(1, '../')
from src.backbone import Backbone
from src.RPN import RPN

class generalized_faster_rcnn(nn.Module):
	"""docstring for generalized_faster_rcnn"""
	def __init__(self, cfg):
		super(generalized_faster_rcnn, self).__init__()
		self.cfg = cfg
		self.backbone_obj = Backbone(self.cfg)
		self.rpn_model = RPN(self.backbone_obj.out_channels, self.cfg)
		
	def forward(self, image):

		"""
			input:
			image: Image has to be of shape Bs x C x H x W
		"""

		## Forward pass for main class
		
		## Forward pass through backbone. Getting feature maps
		feature_map = self.backbone_obj(image)
		rpn_output = self.rpn_model(feature_map)
		
		return rpn_output, feature_map
