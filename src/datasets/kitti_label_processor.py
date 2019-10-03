"""
Transform ground truth loaded through 
pytorch dataloader as arrays for Nuscenes dataset.

"""


import torch
import numpy as np 


def process_labels(cfg, ground_truth):

	"""
		Processes ground truth from kitti dataset
	"""

	## Total ground truth objects in an image
	gt_objects = len(ground_truth[0]) ## It's a batch

	bbox = np.zeros((gt_objects, 4), dtype=np.float64) ## Initializing boudning box
	labels = np.zeros((gt_objects, 1), dtype=np.int16)

	for i in range(gt_objects):
		
		x1 = ground_truth[0][i]['bbox'][0]
		y1 = ground_truth[0][i]['bbox'][1]
		x2 = ground_truth[0][i]['bbox'][2]
		y2 = ground_truth[0][i]['bbox'][3]

		bbox[i,:] = get_y1x1y2x2(x1, y1, x2, y2)
		labels[i,0] = 1

	return {'boxes':bbox, 'labels':labels}


def get_y1x1y2x2(x1, y1, x2, y2):

	return [y1, x1, y2, x2] ## This is how it is expected for our implementation