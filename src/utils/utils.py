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
from src.config import Cfg as cfg

def xy_to_wh(boxes):
	trans = []
	for i in boxes:
		trans.append([i[1], i[0], i[3], i[2]])
	return trans

def draw_bbox(image, bboxes, show_text=False):

	# if len(image.shape)==4:
	# 	image = image[0]
	# if image.shape[2]!=3:
	# 	image = np.transpose(image, (1,2,0))

	# if not isinstance(image, Image):
	# 	image = Image.fromarray(image)
	
	drawer = ImageDraw.Draw(image, mode=None)
	
	for i in bboxes:
		drawer.rectangle(i.cpu().numpy()[:4], outline ='red' ,width=3)
		if show_text:
			drawer.text([i[0], i[1]-10], "{0:.3f}".format(i[4]))
			drawer.text([i[2], i[1]-10], "{0:.3f}".format(i[5]))

	image = image.resize((np.array(image.size)/2).astype(int))
	
	return np.asarray(image)

def image_transform(cfg):

	"""
	Input: 
	cfg: configuration params
	img: Image loaded using matplotlib, numpy array of size H x W x C, in RGB format.
	(Note: if you use opencv to load image, convert it to RGB, as OpenCV works with BGR format)

	Output:
	torch.tensor : CxHxW  nomralised by mean and std of dataset 
	"""

	'''
	### ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor  of shape (C x H x W) in the range [0.0, 1.0] 
	

	'''

	transform = T.Compose([T.ToTensor(),
							T.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)])

	return transform

def toPIL(img: torch.Tensor):
	### T.ToPILImage() Converts an Tensor or numpy array in range [0, 1] with shape of (C x H x W) into an PIL image with range [0,255]
	return T.Compose([T.Normalize( mean=[-mean/std for mean, std in zip(cfg.INPUT.MEAN, cfg.INPUT.STD)], std=[1.0/x for x in cfg.INPUT.STD]), T.ToPILImage()])(img)

def toNumpyImage(img: torch.Tensor):
	return T.Normalize( mean=[-mean/std for mean, std in zip(cfg.INPUT.MEAN, cfg.INPUT.STD)], std=[1.0/x for x in cfg.INPUT.STD])(img).mul(255).numpy().astype('uint8')

def tb_logger(images, tb_writer, boxes=None, rpn_proposals=None, name="Image"):

	image = toPIL(images[2].cpu())
	proposal_locs = rpn_proposals[2].proposal_boxes[:50].tensor
	# pred_locs = boxes[2].pred_boxes.tensor

	proposal_img = draw_bbox(image, proposal_locs)
	# prediction_img = draw_bbox(image, pred_locs)	
		

	image = image.resize((np.array(image.size)/2).astype(int))
	# image_grid = np.concatenate([image ,proposal_img, prediction_img], axis = 1)
	image_grid = np.concatenate([image ,proposal_img], axis = 1)

	tb_writer.add_image(name, image_grid, dataformats='HWC')