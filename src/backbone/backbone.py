'''
Backbone class for Faster-RCNN.
TODO: Make it usable for RetinaNet
'''

import torch
import torchvision
import torch.nn as nn


## TODO: Make it GPU compatible later! .cuda() and .parallel() and so on. 

class Backbone(object):
	"""docstring for Backbone"""
	def __init__(self, stop_layer = None):
		super(Backbone, self).__init__()
		
		self.stop_layer = stop_layer

	def VGG16_backbone(self, image):
		
		'''
		It's a vgg16() backbone. 
		Inputs:
		image: A single image(torch tensor) of size BZxCxHxW, BS = batchsize, C = channels, H = height, W = width

		Output:
		feature_map: Torch tensor of size BSxFcxFhxFw, BS = batchsize, Fc = feature map channels, Fh = feature map height, Fw = feature map width

		'''


		## Using models from pytorch vision repository, more to come! 
		vgg16 = torchvision.models.vgg16(pretrained=True)

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

		## Loading resnet model
		resnet18 = torchvision.models.resnet18(pretrained=True):
		return self.resnet_forward_pass(image, resnet18)

	def resnet34_backbone(self, image):

		## Loading resnet model
		resnet34 = torchvision.models.resnet34(pretrained=True):
		return self.resnet_forward_pass(image, resnet34)

	def resnet50_backbone(self, image):

		## Loading resnet model
		resnet50 = torchvision.models.resnet50(pretrained=True):
		return self.resnet_forward_pass(image, resnet50)

	def resnet101_backbone(self, image):

		## Loading resnet model
		resnet101 = torchvision.models.resnet101(pretrained=True):
		return self.resnet_forward_pass(image, resnet101)

	def resnet152_backbone(self, image):

		## Loading resnet model
		resnet152 = torchvision.models.resnet152(pretrained=True):
		return self.resnet_forward_pass(image, resnet152)


	def VGG11_backbone(self, image):

		raise NotImplementedError

	def VGG19_backbone(self, image):

		raise NotImplementedError	

	def resnet_forward_pass(self, image, model):


		## resnet forward pass until intermediate layer only!
		## Check resnet.py in torchvision/models/ to understand this implementation

		## TODO: Give options to stop forward pass at layer of choice!
		## Just pass a single integer, layer number, where we wish to stop. 
		x = model.conv1(image)
		x = model.bn1(x)
		x = model.relu(x)
		x = model.maxpool(x)

		## If we wish to pass through all the layers
		if self.stop_layer is None:
			x = model.layer1(x)
			x = model.layer2(x)
			x = model.layer3(x)
			x = model.layer4(x)
			return x 

		## If we wish to stop at first major layer
		if self.stop_layer is 1:
			x = model.layer1(x)
			return x

		## If we wish to stop at second major layer
		if self.stop_layer is 2:
			x = model.layer1(x)
			x = model.layer2(x)
			return x
		
		## If we wish to stop at third major layer
		if self.stop_layer is 3:
			x = model.layer1(x)
			x = model.layer2(x)
			x = model.layer3(x)
			return x
		
