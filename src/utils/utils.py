import torch
import os
import sys
import numpy as np
import math
import argparse
from PIL import Image
import matplotlib.image as mpimg ## To load the image
from torch import optim
import os.path as path


def get_actual_coords(prediction, anchors):

	for i in np.arange(prediction['bbox_class'].size()[1]):
		if prediction['bbox_class'][0,i,:][1].item() < 0.8:
			prediction['bbox_pred'][0,i,:] = 0

	prediction['bbox_pred'] = prediction['bbox_pred'].detach().cpu().numpy()
	prediction['bbox_pred'] = prediction['bbox_pred'].reshape([prediction['bbox_pred'].shape[1], prediction['bbox_pred'].shape[2]])


	y_c = prediction['bbox_pred'][:,0]*(anchors[:,2] - anchors[:,0]) + anchors[:,0] + 0.5*(anchors[:,2] - anchors[:,0])
	x_c = prediction['bbox_pred'][:,1]*(anchors[:,3] - anchors[:,1]) + anchors[:,1] + 0.5*(anchors[:,3] - anchors[:,1])
	h = np.exp(prediction['bbox_pred'][:,2])*(anchors[:,2] - anchors[:,0])
	w = np.exp(prediction['bbox_pred'][:,3])*(anchors[:,3] - anchors[:,1])

	x1 = x_c - w/2.0
	y1 = y_c - h/2.0

	bbox_locs_xy = np.vstack((x1, y1, x1+w, y1+h)).transpose() ## Final locations of the anchors

	# print(type(prediction['bbox_pred']), prediction['bbox_pred'].shape, anchors.shape)
	return bbox_locs_xy

def check_validity(x1,y1,w,h, img_w, img_h):
	
	## bottom corner
	x2 = x1 + w
	y2 = y1 + h

	if (x1 > 0 and x2 < img_w) and (y1 > 0 and y2 < img_h):
		return True
	else:
		return False


def xy_to_wh(x1, y1, x2, y2):
	return (x1, y1, x2-x1, y2-y1)

