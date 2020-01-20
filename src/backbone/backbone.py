'''
Backbone class for Faster-RCNN.
'''

import torch
import torchvision
import torch.nn as nn

from .resnet import *

class Backbone(nn.Module):
	"""docstring for Backbone"""
	def __init__(self, cfg):
		super(Backbone, self).__init__()
		
		self.model_name = cfg.BACKBONE.MODEL_NAME

		if "resnet" in self.model_name:
			self.stop_layer = cfg.BACKBONE.RESNET_STOP_LAYER ## used for resnets only

		models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101}

		self.model = models[self.model_name](self.stop_layer, pretrained=True)
		
		if self.stop_layer==3:
			self.stride = 16
		elif self.stop_layer==4:
			self.stride = 32

		key =  'layer' + str(self.stop_layer) + '.1.conv1.weight'
		self.out_channels = self.model.state_dict()[key].shape[1] ## Number of output channels, to be used for RPN

	## Forward pass 
	def forward(self, image):
		return self.model(image)

