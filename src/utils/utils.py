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

def draw_bbox(image, instances):
	class_labels = cfg.INPUT.LABELS_TO_TRAIN	
	drawer = ImageDraw.Draw(image, mode=None)
	# print(instances[20:40])
	for instance in instances:
		if instance.has("proposal_boxes"):
			box = instance.proposal_boxes.tensor.cpu().numpy()[0]
			drawer.rectangle(box, outline ='red' ,width=3)
		elif instance.has("pred_boxes"):
			box = instance.pred_boxes.tensor.cpu().numpy()[0]
			drawer.rectangle(box, outline ='red' ,width=3)
		if instance.has("scores") & instance.has("pred_classes"):
			drawer.text([box[0], box[1]-10],"{}: {:.2f}%".format(class_labels[instance.pred_classes.cpu().numpy()], 
				instance.scores.cpu().numpy()), outline='green')
		if instance.has("pred_sigma"):
			sigma = math.sqrt(instance.pred_sigma)
			drawer.ellipse([box[0]-2*sigma[i, 0], box[1]-2*sigma[i, 1], box[0]+2*sigma[i, 0], box[1]+2*sigma[i,1]], outline='blue', width=3)
			drawer.ellipse([box[2]-2*sigma[i, 2], box[3]-2*sigma[i, 3], box[2]+2*sigma[i, 2], box[3]+2*sigma[i,3]], outline='blue', width=3)

	image = image.resize((np.array(image.size)/1.5).astype(int))
	
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

def tb_logger(images, tb_writer, rpn_proposals=None, instances=None, name="Image"):

	image = toPIL(images[2].cpu())

	image_grid = image.resize((np.array(image.size)/1.5).astype(int))

	if rpn_proposals:
		proposal_locs = rpn_proposals[2].proposal_boxes[:50].tensor.cpu().numpy()
		proposal_img = draw_bbox(image.copy(), proposal_locs)

		image_grid = np.concatenate([image_grid, proposal_img], axis=1)
	
	if instances:
		pred = instances[2].pred_boxes.tensor.cpu().numpy()
		if instances[2].has("pred_sigma"):
			sigma = instances[2].pred_sigma.cpu().numpy()
		else:
			sigma=None
		img_cls = instances[2].pred_classes.cpu().numpy()
		
		prediction_img = draw_bbox(image.copy(), pred, img_cls, sigma)
		image_grid = np.concatenate([image_grid, prediction_img], axis=1)	

	tb_writer.add_image(name, image_grid, dataformats='HWC')


def disk_logger(images, direc, rpn_proposals=None, instances=None, image_paths=None):

	images = images.cpu()
	pil_images = []

	for image, rpn_proposal, instance, path in zip(images, rpn_proposals, instances, image_paths):
		image = toPIL(image)

		image_grid = image.resize((np.array(image.size)/1.5).astype(int))

		if rpn_proposal:
			proposal_img = draw_bbox(image.copy(), rpn_proposal)
			image_grid = np.concatenate([image_grid, proposal_img], axis=1)
		
		if instance:			
			prediction_img = draw_bbox(image.copy(), instance)
			image_grid = np.concatenate([image_grid, prediction_img], axis=1)

		Image.fromarray(image_grid).save(direc+"/"+path[-10:], "PNG")
		print("{} written to disk".format(path[-10:]))

