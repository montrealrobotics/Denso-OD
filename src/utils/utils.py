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
from torchvision import transforms as T

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
		drawer.rectangle(i.cpu().numpy()[:4], outline ='red' ,width=3)
		if show_text:
			drawer.text([i[0], i[1]-10], "{0:.3f}".format(i[4]))
			drawer.text([i[2], i[1]-10], "{0:.3f}".format(i[5]))

	return np.asarray(image), image	


def image_transform(cfg):

	"""
	Input: 
	cfg: configuration params
	img: Image loaded using matplotlib, numpy array of size H x W x C, in RGB format.
	(Note: if you use opencv to load image, convert it to RGB, as OpenCV works with BGR format)

	Output:
	torch tensor of size 1xCxHxW
	"""

	'''
	### ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor  of shape (C x H x W) in the range [0.0, 1.0] 

	'''

	transform = T.Compose([T.ToTensor(),
							T.Normalize(mean=list(cfg.INPUT.MEAN), std=list(cfg.INPUT.STD))])

	inverse_transorf = T.Compose([T.Normalize( mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]), T.ToPILImage()])

	return transform, inverse_transorf









