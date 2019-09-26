"""
Compute loss for Region proposal networks(Not supported for RetinaNet yet)
"""

import torch
import sys

class RPNLoss(torch.nn.Module):
	"""docstring for RPNLoss"""
	def __init__(self, cfg):
		super(RPNLoss, self).__init__()

		## Criterion for classification
		self.class_criterion =  torch.nn.CrossEntropyLoss()

		self.total_anchors = None
		self.class_loss = None
		self.reg_loss = None
		self.cfg = cfg

		## Deterministic object detection
		if self.cfg.TRAIN.TRAIN_TYPE == "deterministic":
			self.reg_criterion = torch.nn.SmoothL1Loss(reduction='mean')

	## TODO: get it to work for a batch. 
	def forward(self, prediction, target, valid_indices):

		"""
		Input: 
		prediction: A dictionary with three keys("bbox_pred", "bbox_uncertainty_pred", "bbox_class")
		target: A dictionary with two keys("gt_bbox", "gt_anchor_label")
		valid_indices: A set of valid indices for valid anchors, python list

		If there are N anchors, then,
			1. Shape of prediction['bbox_pred'] = Bs x N x 4
			2. Shape of prediction['bbox_uncertainty_pred'] = Bs x N x 4
			3. Shape of prediction['bbox_class'] = Bs x N x 2
			4. Shape of target['gt_bbox'] = Bs x N x 4
			5. Shape of target['gt_anchor_label'] = Bs x N 

		"""

		self.total_anchors = prediction['bbox_pred'].shape[1]

		self.class_loss = self.get_classification_loss(prediction, target, valid_indices)
		self.reg_loss = self.get_regression_loss(prediction, target, valid_indices)

		# print("Classification and regression losses are: ", self.class_loss.item(), self.reg_loss.item())
		return self.class_loss + self.reg_loss

	
	def get_classification_loss(self, prediction, target, valid_indices):


		'''
			Compute classification loss for RPN, between +ve and -ve anchor
		'''

		return self.class_criterion(prediction['bbox_class'][0][valid_indices], target['gt_anchor_label'][0][valid_indices])

	def get_regression_loss(self, prediction, target, valid_indices):

		'''	
			Compute regression loss using loss attenuation formulation

		'''

		if self.cfg.TRAIN.TRAIN_TYPE == "deterministic":
			# print("Calculating deterministic loss!")
			pos_indices = []
			## Finding anchors with positive indices
			# print("Length of valid_indices is: ", valid_indices.shape)
			for valid_index in valid_indices[0]:
				if target['gt_anchor_label'][0][valid_index].item() == 1:
					# print(valid_index)
					pos_indices.append(valid_index)

				# sys.exit(0)
			if len(pos_indices)*2 != len(valid_indices[0]):
				print("Total and pos anchors are: ", len(valid_indices[0]), len(pos_indices))
			# print("length of total valid anchors is: ", len(valid_indices[0]))
			reg_loss = self.reg_criterion(prediction['bbox_pred'][0][pos_indices], target['gt_bbox'][0][pos_indices])
			return reg_loss

		else:
			self.pos_anchor_loss = torch.zeros(1).type(self.cfg.DTYPE.FLOAT)
			self.neg_anchor_loss = torch.zeros(1).type(self.cfg.DTYPE.FLOAT)
			self.pos_anchors = torch.zeros(1).type(self.cfg.DTYPE.FLOAT)
			self.neg_anchors = torch.zeros(1).type(self.cfg.DTYPE.FLOAT)

			for valid_index in valid_indices[0]:

				## if anchor is positive, compute intelligent robust regression loss
				if target['gt_anchor_label'][0][valid_index].item() == 1:
					self.pos_anchor_loss +=  (0.5*(((prediction['bbox_pred'][0][valid_index] - target['gt_bbox'][0][valid_index]).pow(2))/(1e-3 + prediction['bbox_uncertainty_pred'][0][valid_index].pow(2))) + \
												0.5*torch.log(1e-3 + prediction['bbox_uncertainty_pred'][0][valid_index].pow(2))).sum()
					self.pos_anchors += 1
				## Encourage high uncertainty for negative anchors
				else: 
					self.neg_anchor_loss += (1.0/(1e-3 + prediction['bbox_uncertainty_pred'][0][valid_index].pow(2))).sum()
					self.neg_anchors += 1

			if self.pos_anchors == 0 or self.neg_anchors == 0:
				return None ## when you can't compute regression loss, let it go as 0

			else:
				reg_loss = (self.pos_anchor_loss/self.pos_anchors)+ (self.neg_anchor_loss/self.neg_anchors)

			return reg_loss
