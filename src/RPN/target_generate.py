
'''
Class to generate targets for 
Region proposal network. 


'''

import torch
import sys
import numpy as np
from .anchors import anchor_generator


class RPN_targets(object):
	"""Generates target for region proposal network.(Useful for training)"""
	def __init__(self, cfg):
		super(RPN_targets, self).__init__()
		self.anchor_generator_obj = anchor_generator()
		self.cfg = cfg

	def get_targets(self, image, feature_map, targets):


		"""
		Inputs: 
		Image: Image or batch of images as torch tensor
		feature_map: feature map output of backbone network
		Targets: Ground truth labels and bonding boxes corresponding to the image

		"""
	
		anchors = self.anchor_generator_obj.get_anchors(image, feature_map, self.cfg) ## Nx4 numpy array
		# print(anchors.shape)
		orig_anchors = anchors

		## Some of these anchors may not be valid
		## Let's get indices of anchors which are inside the image
		im_height = image.size()[2]
		im_width = image.size()[3]

		inside_indices = np.where((anchors[:,0] >= 0) &
							 (anchors[:,1] >= 0) &
							 (anchors[:,2] <= im_height) &
							 (anchors[:,3] <= im_width))[0]

		# print(len(inside_indices))
		## Constructing an array holding valid anchor boxes
		## These anchors basically fall inside the image
		inside_anchors = anchors[inside_indices]

		'''
		Each anchor will either be positive or negative. Hence, we are creating 
		an array of length same as number of valid anchors, and we will assign them
		either 0(negative anchor) or 1(positive anchor) or -1(not valid)
		'''

		anchor_labels = np.empty((len(inside_indices), ), dtype=np.int32)
		anchor_labels.fill(-1)

		ious = self.compute_anchor_iou(inside_anchors, targets['boxes'])

		'''
		Now comes the part where we assign labels to anchors. 

		Positive anchors: 
			1. The anchors with highest IoU with the ground truth objects
			2. Anchor with IoU > 0.7 with the ground truth object.

		Negative anchors:
			1. All the anchors whose IoU with all the ground truth objects is lesser than 0.3,
			   are negative anchors. 
		'''


		## Finding highest IoU for each gt_box, and the corresponding anchor box
		## Contains M indices in range [0, N], for M objects
		gt_argmax_ious = ious.argmax(axis=0) 
		# print(len(gt_argmax_ious))

		## saving maximum ious, for all the ground truth objects, gives the maximun IoU
		## Contains M IoUs, for M ground truth objects
		gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] 
		
		"""
			This solves a very important bug. 
			A lot of times, it's possible that due to object's faulty
			annotation around the corners, we may not get proper maximum
			IoU, we may end up with 0 IoU, which can mess up the calculation
			ahead, so we identify such a case and don't let our network train 
			on that. Instead, we leave the function(effectively raising an exception
			in a training script, which results in moving to the next image)
		"""
		if np.sum(gt_max_ious < 1e-9) > 0:
			return

		## Finding maximum IoU for each anchor and its corresponding GT box
		## Contains object label with highest IoU
		## Contains N indices in range [0, M]
		argmax_ious = ious.argmax(axis=1) 
		# print(argmax_ious)

		## saving maximum IoU for each anchor 
		## Contains N IoUs, for N anchors. Highest IoU for each anchor
		max_ious = ious[np.arange(ious.shape[0]),argmax_ious] 

		## we gotta find all the anchor indices with gt_max_ious
		## Multiple anchors could have highest IoUs which we discovered earlier. 
		## Contains ID of anchors with highest IoUs
		# print(ious.shape, gt_max_ious)

		gt_argmax_ious = np.where(ious == gt_max_ious)[0]
		# print(gt_argmax_ious)
		

		'''
		Let's assign labels! Important
		'''

		pos_anchor_iou_threshold = 0.7
		neg_anchor_iou_threshold = 0.3


		## IF max_iou for and anchor is lesser than neg_anchor_iou_threshold, it's a negative anchor.
		anchor_labels[max_ious < neg_anchor_iou_threshold] = 0		


		# print(np.max(max_ious))
		## All the anchors with highest IoUs get to be positive anchors
		anchor_labels[gt_argmax_ious] = 1
		# print(len(gt_argmax_ious))
		
		# print(len(anchor_labels))
		## All the anchors with iou greater than pos_anchor_iou_threshold deserve to be a positive anchor
		anchor_labels[max_ious >= pos_anchor_iou_threshold] = 1

		# print(len(anchor_labels[max_ious < neg_anchor_iou_threshold]))

		## We don't use all anchors to compute loss. We sample negative and positive anchors 
		## in 1:1 ratio to avoid domination of negative anchors.

		## Ratio of positive and negative anchors
		pos_to_neg_ratio = 0.5
		num_of_anchor_samples = 256 ## Total number of anchors

		n_pos = int(num_of_anchor_samples*pos_to_neg_ratio) ## Number of positive anchors
		n_neg = num_of_anchor_samples - n_pos          ## Number of negative anchors

		'''
		Sampling positive and negative anchors.

		Sometimes, it possible that number of positive anchors
		are lesser than number of samples required. In that case, 
		we keep all of them for our purpose. But if we have lesser 
		number of positive proposals than required, we also make
		number of negative proposals = number of positive proposals

		'''

		pos_anchor_indices = np.where(anchor_labels == 1)[0] ## Indices with positive label
		neg_anchor_indices = np.where(anchor_labels == 0)[0] ## Indices with negaitve label
		# print(len(anchor_labels), np.sum(anchor_labels == 0), np.sum(anchor_labels == 1), np.sum(anchor_labels == -1))
		# print("Number of negative anchors are: ", len(neg_anchor_indices))

		if len(pos_anchor_indices) > n_pos:
			disable_index = np.random.choice(pos_anchor_indices, size=(len(pos_anchor_indices) - n_pos), replace=False)
			anchor_labels[disable_index] = -1

		### Important!!! ###
		'''
		If we have lesser number of positive anchors, we keep all of them. 
		But then we also sample lesser number of negative anchors. Because
		we need to keep their ratio proper. 
		'''

		if len(pos_anchor_indices) < n_pos:
			n_neg = len(pos_anchor_indices)
			
		if len(neg_anchor_indices) > n_neg:
			disable_index = np.random.choice(neg_anchor_indices, size=(len(neg_anchor_indices) - n_neg), replace=False)
			anchor_labels[disable_index] = -1

		print( np.sum( anchor_labels == 1) , np.sum( anchor_labels == 0))

		'''
		Labels have already been assigned to the anchors, now we need to
		assign locations to anchor boxes. Assign them the ground truth object
		with maximum IOU. 
		'''

		## We have N valid achors, for each valid anchor, we have a corresponding 
		## groundtruth bounding box. We compute them as below. For all anchors, 
		## we need del_x, del_y, del_h, del_w, where (x,y) and (h,w) are center
		## of the anchor and height, width respectively. 
		bbox = targets['boxes']
		max_iou_bbox = bbox[argmax_ious]

		## Getting anchor centers and anchor dimensions(height, width)!
		anchor_height = inside_anchors[:,2] - inside_anchors[:,0] ## N x 1: height of all N anchors
		anchor_width = inside_anchors[:,3] - inside_anchors[:,1]  ## N x 1: width of all N anchors
		anchor_ctr_y = inside_anchors[:,0] + 0.5 * anchor_height      ## N x 1: y-coordinates of all Anchor centers
		anchor_ctr_x = inside_anchors[:,1] + 0.5 * anchor_width      ## N x 1: x-coordinates of all Anchor centers

		## Getting ground truth BB centers and dimensions(height, width)

		## N x 1: height of all N groundtruth bounding boxes for N anchors
		base_height = max_iou_bbox[:,2] - max_iou_bbox[:,0] 
		## N x 1: height of all N groundtruth bounding boxes for N anchors
		base_width = max_iou_bbox[:,3] - max_iou_bbox[:,1]  
		## N x 1: y-coordinates of the center of all N groundtruth bounding boxes for N anchors
		base_ctr_y = max_iou_bbox[:,0] + 0.5 * base_height      
		## N x 1: x-coordinates of the center of all N groundtruth bounding boxes for N anchors
		base_ctr_x = max_iou_bbox[:,1] + 0.5 * base_width

		'''
		Using above information to find locations as required by Faster R-CNN
		'''

		eps = np.finfo(anchor_height.dtype).eps
		anchor_height = np.maximum(anchor_height, eps)
		anchor_width = np.maximum(anchor_width, eps)

		'''
		As required by Faster R-CNN
		'''
		dy = (base_ctr_y - anchor_ctr_y) / anchor_height
		dx = (base_ctr_x - anchor_ctr_x) / anchor_width
		dh = np.log(base_height / anchor_height)
		dw = np.log(base_width / anchor_width)

		anchor_locs = np.vstack((dy, dx, dh, dw)).transpose() ## Final locations of the anchors

		anchor_labels_final = np.empty((len(anchors), ), dtype = anchor_labels.dtype)
		anchor_labels_final.fill(-1)
		anchor_labels_final[inside_indices] = anchor_labels

		anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
		anchor_locations.fill(0)
		anchor_locations[inside_indices, :] = anchor_locs

		return anchor_locations, anchor_labels_final, orig_anchors

	def compute_anchor_iou(self, valid_anchor_boxes, ground_truth_objects):

		'''
		Let's say we have N valid anchors and M objects in an image. 
		We compute IOU of all the valid anchors with all the ground-truth objects.
		So we will have an N x M array storing IOUs
		
		Inputs: 
		valid_anchor_boxes: N x 4 array containing anchor co-ordinates
		ground_truth_objects: M x 4 array containing ground-truth object co-ordinates
		
		Output:
		ious: N x M arrays containing IOUs between each anchor and ground truth object
		
		'''
		
		## Number of anchors
		N = valid_anchor_boxes.shape[0]

		## Number of Ground truth objects
		M = ground_truth_objects.shape[0]
		
		## Initializing array for IOU
		ious = np.empty((N, M), dtype=np.float32)
		ious.fill(0)
		
		## Let's start computing!!!!!!!
		for anchor_index, anchor_coord in enumerate(valid_anchor_boxes):
			ya1, xa1, ya2, xa2 = anchor_coord
			anchor_area = (ya2 - ya1)*(xa2 - xa1)
			
			for gt_object_index, gt_coord in enumerate(ground_truth_objects):
				yb1, xb1, yb2, xb2 = gt_coord
				gt_object_area = (yb2 - yb1)*(xb2 - xb1)
				
				## getting the inner area
				inter_x1 = max([xb1, xa1])
				inter_y1 = max([yb1, ya1])
				inter_x2 = min([xb2, xa2])
				inter_y2 = min([yb2, ya2])
				
			
				if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
					inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
					iou = inter_area/(anchor_area + gt_object_area - inter_area)
					
				else:
					iou = 0
		
				ious[anchor_index, gt_object_index] = iou
			
			
		return ious
