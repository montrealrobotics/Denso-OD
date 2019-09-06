"""
Transform ground truth loaded through 
pytorch dataloader as arrays.

"""


import torch
import numpy as np 


def process_labels(ground_truth):

	"""
		Processes ground truth from coco style 
		annotation to our required format
	"""

	## Total ground truth objects in an image
	gt_objects = len(ground_truth)

	bbox = np.zeros((gt_objects, 4), dtype=np.float64) ## Initializing boudning box
	labels = np.zeros((gt_objects, 1), dtype=np.int16)

	for i in range(gt_objects):

		x = ground_truth[i]['bbox'][0].item()
		y = ground_truth[i]['bbox'][1].item()
		w = ground_truth[i]['bbox'][2].item()
		h = ground_truth[i]['bbox'][3].item()

		bbox[i,:] = get_y1x1y2x2(x, y, w, h)
		labels[i,0] = ground_truth[i]['category_id'].item()

	return {'boxes':bbox, 'labels':labels}


def get_y1x1y2x2(x, y, w, h):

	x1 = x
	y1 = y
	x2 = x + w
	y2 = y + h

	return [y1, x1, y2, x2] ## This is how it is expected for our implementation