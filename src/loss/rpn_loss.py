"""
Compute loss for Region proposal networks(Not supported for RetinaNet yet)
"""

import torch
import torch.nn.functional as F
import sys
from src.config import Cfg as cfg


def smooth_l1_loss(input, target, beta: float, reduction: str = "none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_objectness_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
            its matched ground-truth box.
        pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = gt_objectness_logits == 1
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], smooth_l1_beta, reduction="sum"
    )

    valid_masks = gt_objectness_logits >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness_logits[valid_masks],
        gt_objectness_logits[valid_masks].to(torch.float32),
        reduction="sum",
    )
    return objectness_loss, localization_loss

class RPNErrorLoss(torch.nn.Module):
	"""docstring for RPNLoss"""
	def __init__(self, cfg):
		super(RPNErrorLoss, self).__init__()

		## Criterion for classification
		self.class_criterion =  torch.nn.CrossEntropyLoss()
		self.total_anchors = None
		self.class_loss = None
		self.reg_loss = None
		self.cfg = cfg

		## new loss of RPN
		self.reg_criterion = torch.nn.SmoothL1Loss(reduction='mean')	

	## TODO: get it to work for a batch. 
	def forward(self, prediction, target, valid_indices):

		"""
		Input: 
		prediction: A dictionary with three keys("bbox_pred", "bbox_uncertainty_pred", "bbox_class")
		target: A dictionary with two keys("gt_bbox", "gt_anchor_label")
		valid_indices: A set of valid indices for valid anchors, python list

		If there are N anchors, then,
			1. Shape of prediction[0] = Bs x N x 4
			2. Shape of prediction[2] = Bs x N x 4
			3. Shape of prediction[1] = Bs x N x 2
			4. Shape of target['gt_bbox'] = Bs x N x 4
			5. Shape of target['gt_anchor_label'] = Bs x N 

		"""

		self.total_anchors = prediction[0].shape[1]

		self.class_loss = cfg.TRAIN.CLASS_LOSS_SCALE*self.get_classification_loss(prediction, target, valid_indices)
		self.reg_loss_bbox, self.error_loss_bbox = self.get_regression_loss(prediction, target, valid_indices)
		self.reg_loss_bbox = cfg.TRAIN.SMOOTHL1LOSS_SCALE*self.reg_loss_bbox

		# print("Classification and regression losses are: ", self.class_loss.item(), self.reg_loss.item())
		return self.class_loss, self.reg_loss_bbox, self.error_loss_bbox

	
	def get_classification_loss(self, prediction, target, valid_indices):


		'''
			Compute classification loss for RPN, between +ve and -ve anchor
		'''

		return self.class_criterion(prediction[1][0][valid_indices], target['gt_anchor_label'][0][valid_indices])



	def get_regression_loss(self, prediction, target, valid_indices):

		'''	
			train with a newly proposed loss
		'''

		## Calculating smoothL1Loss for bounding box regression
		pos_indices = []
		neg_indices = []
		## Finding anchors with positive indices
		# print("Length of valid_indices is: ", valid_indices.shape)
		for valid_index in valid_indices[0]:
			if target['gt_anchor_label'][0][valid_index].item() == 1:
				# print(valid_index)
				pos_indices.append(valid_index)
			else:
				neg_indices.append(valid_index)
		
		# if len(pos_indices)*2 != len(valid_indices[0]):
			# print("Total and pos anchors are: ", len(valid_indices[0]), len(pos_indices))
		# print("length of total valid anchors is: ", len(valid_indices[0]))
		reg_loss = self.reg_criterion(prediction[0][0][pos_indices], target['gt_bbox'][0][pos_indices])
		


		## Calculating loss over the error
		pred_clone = prediction[0][0][pos_indices].clone().detach()
		target_clone = target['gt_bbox'][0][pos_indices]

		error_actual = pred_clone - target_clone

		## Whatever is coming out of RPN for uncertainty is basically error predicted
		error_predicted = prediction[2][0][pos_indices]

		error_loss = self.reg_criterion(error_predicted, error_actual)

		return reg_loss, error_loss 







