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

	def __init__(self, in_channels, n_anchors):
		super(RPN, self).__init__()
		self.in_channels = in_channels  ## Number of channels in feature map
		self.out_channels = 512 		## Number of output channels from first layer of RPN
		self.n_anchors = n_anchors		## Number of anchors per location

		## Layer 1
		self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
		# self.conv1.weight.data.normal(0, 0.01)
		# self.conv1.bias.data.zero_()
		
		## Regression layer
		self.reg_layer = nn.Conv2d(self.out_channels, self.n_anchors*4, 1, 1, 0)
		# self.reg_layer.weight.data.normal(0,0.01)
		# self.reg_layer.bias.data.zero_()

		## classification layer
		self.classification_layer = nn.Conv2d(self.out_channels, self.n_anchors*2, 1, 1, 0)
		# self.classification_layer.weight.data.normal(0, 0.01)
		# self.classification_layer.bias.data.zero_()

		## Uncertainty layer
		self.uncertain_layer = nn.Conv2d(self.out_channels, self.n_anchors*4, 1, 1, 0)

	def forward(self, feature_map):

		## forward pass
		result = {}

		## Layer 1 forward pass
		x = self.conv1(feature_map)

		## Output of regression layer 
		result['regression'] = self.reg_layer(x)

		## Output of classification layer
		result['classification'] = self.classification_layer(x)

		## Output of uncertainty layer
		result['uncertainty'] = self.uncertain_layer(x)

		return self.reshape_output(result)

	def reshape_output(self, result):

		'''
		This reshapes the output of the network.
		Example: 
		If output of regression layer is 8x36x18x18, the final 
		output will be of size 8x2916x4

		If output of regression layer is 8x18x18x18, the final 
		output will be of size 8x2916x2(2916 = total number of anchors per image, 8 = batchsize)

		'''
		## Make this better, but ok for now!

		final_output = {}

		final_output['regression'] = result['regression'].view((result['regression'].size()[0],
															int(result['regression'].size()[1]*result['regression'].size()[2]*result['regression'].size()[3]/4),
															4))

		final_output['uncertainty'] = result['uncertainty'].view((result['uncertainty'].size()[0],
															int(result['uncertainty'].size()[1]*result['uncertainty'].size()[2]*result['uncertainty'].size()[3]/4),
															4))

		final_output['classification'] = result['classification'].view((result['classification'].size()[0],
															int(result['classification'].size()[1]*result['classification'].size()[2]*result['classification'].size()[3]/2),
															2))
		return final_output