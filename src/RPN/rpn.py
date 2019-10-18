'''
Region proposal network baseclass.
'''

import torch
import torch.nn as nn

class RPN(nn.Module):
	"""docstring for RPN"""

	'''
	TODO: Could add more layers and experiment around as and when required. Just pass the config for all this.
	'''

	def __init__(self, in_channels, cfg):
		super(RPN, self).__init__()
		self.in_channels = in_channels  ## Number of channels in feature map
		self.out_channels = cfg.RPN.OUT_CHANNELS  		## Number of output channels from first layer of RPN
		self.rpn_channels = cfg.RPN.LAYER_CHANNELS		## Channels in RPN layers
		self.n_anchors = cfg.RPN.N_ANCHORS_PER_LOCATION		## Number of anchors per location

		## Layer 1
		self.conv1 = nn.Conv2d(self.in_channels, self.rpn_channels[0], 3, 1, 1)
		self.conv1.weight.data.normal_(cfg.RPN.CONV_MEAN, cfg.RPN.CONV_VAR)
		self.conv1.bias.data.fill_(cfg.RPN.BIAS)
		# nn.init.xavier_uniform(self.conv1.weight)
		self.eLU1 = nn.ELU(alpha = cfg.RPN.ACTIVATION_ALPHA)

		## Regression layer
		self.reg_layer = nn.Conv2d(self.rpn_channels[0], self.n_anchors*4, 1, 1, 0)
		self.reg_layer.weight.data.normal_(cfg.RPN.CONV_MEAN, cfg.RPN.CONV_VAR)
		self.reg_layer.bias.data.fill_(cfg.RPN.BIAS)
		# nn.init.xavier_uniform(self.reg_layer.weight)
		self.eLU_reg = nn.ELU(alpha = cfg.RPN.ACTIVATION_ALPHA)

		## classification layer
		self.classification_layer = nn.Conv2d(self.rpn_channels[0], self.n_anchors*2, 1, 1, 0)
		# self.softmax_classification = nn.Softmax(dim=2)
		self.classification_layer.weight.data.normal_(cfg.RPN.CONV_MEAN, cfg.RPN.CONV_VAR)
		self.classification_layer.bias.data.fill_(cfg.RPN.BIAS)
		# nn.init.xavier_uniform(self.classification_layer.weight)

		## Uncertainty layer
		self.uncertain_layer = nn.Conv2d(self.rpn_channels[0], self.n_anchors*4, 1, 1, 0)
		self.uncertain_layer.weight.data.normal_(cfg.RPN.UNCERTAIN_MEAN, cfg.RPN.UNCERTAIN_VAR)
		self.uncertain_layer.bias.data.fill_(cfg.RPN.UNCERTAIN_BIAS)	## Initialize with high values to avoid NaNs
		# nn.init.xavier_uniform(self.uncertain_layer.weight)
		# self.softplus =  nn.Softplus(beta = cfg.RPN.SOFTPLUS_BETA, threshold = cfg.RPN.SOFTPLUS_THRESH)
		self.eLU_sigma = nn.ELU(alpha = cfg.RPN.ACTIVATION_ALPHA)


		## Softplus for uncertainty


	def forward(self, feature_map):

		## forward pass
		# result = {}

		## Layer 1 forward pass
		x = self.eLU1(self.conv1(feature_map))

		## Layer 2 forward pass
		# x = self.eLU(self.conv2(x))

		# ## Layer 3 forward pass
		# x = self.eLU(self.conv3(x))

		## Output of regression layer 
		regress_out = self.eLU_reg(self.reg_layer(x))
		regress_out = regress_out.view((regress_out.shape[0], 4, -1))
		regress_out = regress_out.permute(0, 2, 1)
		## Output of classification layer
		# result['classification'] = self.softmax_classification(self.classification_layer(x))
		class_out = self.classification_layer(x)
		class_out = class_out.view((class_out.shape[0], 2, -1))
		class_out = class_out.permute(0, 2, 1)

		## Output of uncertainty layer
		sigma_out = self.eLU_sigma(self.uncertain_layer(x))
		sigma_out = sigma_out.view((sigma_out.shape[0], 4, -1))
		sigma_out = sigma_out.permute(0,2,1)
		
		return (regress_out, class_out, sigma_out)


	def RichardCurve(self, x, low=0, high=1, sharp=0.5):
		"""Applies the generalized logistic function (aka Richard's curve)
		to the input tensor x.

		Args:
			x (torch.Tensor): Input tensor over which the generalized logistic
				function is to be applied (independently over each element)
			low (float): Lower asymptote of the Richard's curve
			high (float): Upper asymptote of the Richard's curve
			sharp (float): Controls the 'sharpness' of the slope for the linear
				region of Richard's curve

		"""
		return low + ((high - low) / (1 + torch.exp(-sharp * x)))


	def reshape_output(self, result):

		'''
		This reshapes the output of the network.
		Example: 
		If output of regression layer is 8x36x18x18, the final 
		output will be of size 8x2916x4

		If output of classification layer is 8x18x18x18, the final 
		output will be of size 8x2916x2(2916 = total number of anchors per image, 8 = batchsize)

		'''
		## Make this better, but ok for now!

		final_output = {}

		final_output['bbox_pred'] = result['regression'].view((result['regression'].size()[0],
															int(result['regression'].size()[1]*result['regression'].size()[2]*result['regression'].size()[3]/4),
															4))

		final_output['bbox_uncertainty_pred'] = result['uncertainty'].view((result['uncertainty'].size()[0],
															int(result['uncertainty'].size()[1]*result['uncertainty'].size()[2]*result['uncertainty'].size()[3]/4),
															4))

		final_output['bbox_class'] = result['classification'].view((result['classification'].size()[0],
															int(result['classification'].size()[1]*result['classification'].size()[2]*result['classification'].size()[3]/2),
															2))
		return final_output

