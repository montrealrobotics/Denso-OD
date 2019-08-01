'''
Backbone class for Faster-RCNN.
TODO: Make it usable for RetinaNet
'''

import torch
import torchvision.models as model
import torch.nn as nn


## TODO: Make it GPU compatible later! .cuda() and .parallel() and so on. 

class Backbone(object):
	"""docstring for Backbone"""
	def __init__(self):
		super(Backbone, self).__init__()
		## Do nothing for now! 

	def VGG16_backbone(self, image):
		
		'''
		It's a vgg16() backbone. 
		Inputs:
		image: A single image(torch tensor) of size BZxCxHxW, BS = batchsize, C = channels, H = height, W = width

		Output:
		feature_map: Torch tensor of size BSxFcxFhxFw, BS = batchsize, Fc = feature map channels, Fh = feature map height, Fw = feature map width

		'''


		## Using models from pytorch vision repository, more to come! 
		vgg16 = model.vgg16(pretrained=True)

		fake_input = image.clone()
		im_height = image.size()[2] ## 0: Batch size, 1: image channels, 2: image height, 3: image width

		## Make it a parameter! To determine when to stop. 
		sub_sample = 16
		req_features = []

		## TODO: Freeze the layers
		for layer in list(vgg16.features):
		    fake_input = layer(fake_input)
		    # print(fake_input.size())		## For debugging! To be removed later.
		    if fake_input.size()[2] < im_height//sub_sample: ## Because final Convolutional feature map size should be less than this
		        break
		    req_features.append(layer)
		    out_channels = fake_input.size()[1]

		## Construct a model with required layers
		frcnn_backbone = nn.Sequential(*req_features)
		feature_map = frcnn_backbone(image)
		return feature_map

	## TODO: Implement for various different BACKBONES like VGG one! 
	def resnet18_backbone(self, image):

		raise NotImplementedError

	def resnet34_backbone(self, image):

		raise NotImplementedError

	def resnet50_backbone(self, image):

		raise NotImplementedError

	def VGG11_backbone(self, image):

		raise NotImplementedError

	def VGG19_backbone(self, image):

		raise NotImplementedError	
