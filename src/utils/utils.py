import torch
import os
import sys
import numpy as np
import math
import argparse
from PIL import Image, ImageDraw
import matplotlib.image as mpimg ## To load the image
from torch import optim
import os.path as path



def get_actual_coords(prediction, anchors):

	for i in np.arange(prediction[1].size()[1]):
		if prediction[1][0,i,:][1].item() < 0.8:
			prediction[0][0,i,:] = 0

	bboxes = prediction[0].detach().cpu().numpy()
	bboxes = bboxes.reshape([bboxes.shape[1], bboxes.shape[2]])
	pred = prediction[1].detach().cpu().numpy()

	y_c = bboxes[:,0]*(anchors[:,2] - anchors[:,0]) + anchors[:,0] + 0.5*(anchors[:,2] - anchors[:,0])
	x_c = bboxes[:,1]*(anchors[:,3] - anchors[:,1]) + anchors[:,1] + 0.5*(anchors[:,3] - anchors[:,1])
	h = np.exp(bboxes[:,2])*(anchors[:,2] - anchors[:,0])
	w = np.exp(bboxes[:,3])*(anchors[:,3] - anchors[:,1])
	prob = pred[0,:,1]
	# print(prob.shape)

	x1 = x_c - w/2.0
	y1 = y_c - h/2.0

	bbox_locs_xy = np.vstack((x1, y1, x1+w, y1+h, prob)).transpose() ## Final locations of the anchors
	# print(type(bboxes), bboxes.shape, anchors.shape)
	return bbox_locs_xy

def check_validity(x1,y1,w,h, img_w, img_h):
	
	## bottom corner
	x2 = x1 + w
	y2 = y1 + h

	if (x1 > 0 and x2 < img_w) and (y1 > 0 and y2 < img_h):
		return True
	else:
		return False


def xy_to_wh(boxes):
	trans = []
	for i in boxes:
		trans.append([i[1], i[0], i[3], i[2]])
	return trans

def draw_bbox(image, bboxes, show_text=False):

	if len(image.shape)==4:
		image = image[0]
	if image.shape[2]!=3:

		image = np.transpose(image, (1,2,0))

	image = Image.fromarray(image)
	drawer = ImageDraw.Draw(image, mode=None)
	
	for i in bboxes:
		drawer.rectangle(i[:4], outline ='red' ,width=3)
		if show_text:
			drawer.text([i[0], i[1]-10], "{0:.3f}".format(i[4]))

	return np.asarray(image), image	








