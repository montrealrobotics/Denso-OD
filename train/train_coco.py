"""
Training RPNs. 
"""


"""
How to run on MILA cluster?

python train.py -dp "/network/tmp1/bhattdha/coco_dataset_new/train2017_modified/" -ap "/network/tmp1/bhattdha/coco_dataset_new/annotations_modified/instances_train2017_modified.json" -mp "/network/tmp1/bhattdha/Denso-models/"

"""

import torch
import os
import sys
import numpy as np
import math
import argparse
import matplotlib.image as mpimg ## To load the image
from torch import optim
import os.path as path
## Inserting path of src directory
sys.path.insert(1, '../')
from src.architecture import FRCNN
from src.config import Cfg as cfg
from src.RPN import anchor_generator, RPN_targets
from src.preprocess import image_transform ## It's a function, not a class. 
from src.datasets import process_coco_labels
from src.loss import RPNLoss
from torchvision import datasets as dset

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--datasetpath", required = True, help="give dataset path")
ap.add_argument("-ap", "--annotationpath", required = True, help="give annotation file path")
ap.add_argument("-mp", "--modelpath", required = True, help="give model directory path")

args = vars(ap.parse_args())
dset_path = args["datasetpath"]
ann_path = args["annotationpath"]
model_dir_path = args["modelpath"]

if not path.exists(dset_path):
	print("Dataset path doesn't exist")
if not path.exists(ann_path):
	print("Annotation path doesn't exist")
if not path.exists(model_dir_path):
	os.mkdir(model_dir_path)

# Setting the seeds
torch.manual_seed(5)
np.random.seed(5)

## setting default variable types
torch.set_default_tensor_type('torch.FloatTensor') 
torch.set_default_dtype(torch.float32)

### use cuda only if it's available and permitted
if torch.cuda.is_available() and not cfg.NO_GPU:
	cfg.USE_CUDA = True

### let's generate the dataset
tranform = image_transform(cfg)
coco_dataset = dset.CocoDetection(dset_path, ann_path, transform= tranform) 
trainloader = torch.utils.data.DataLoader(coco_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE)

# Generate random input
# DONE: replace with actual image later,with vision tranforms(normalization)
# input_image, labels = iter(trainloader).next()
# targets = process_coco_labels(labels)
# print(targets)



frcnn = FRCNN(cfg)
if cfg.TRAIN.FREEZE_BACKBONE:
	for params in frcnn.backbone_obj.parameters():
		params.requires_grad = False

## Initialize RPN params

optimizer = optim.Adam(frcnn.parameters())

checkpoint_path = model_dir_path + 'checkpoint.txt'

if path.exists(checkpoint_path):
	with open(checkpoint_path, "r") as f: 
		model_path = f.readline().strip('\n')

	## Only load if such a model exists
	if path.exists(model_path):

		checkpoint = torch.load(model_path)
		frcnn.load_state_dict(checkpoint['model_state_dict'])


		## TO load the optimizer state with cuda
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.cuda() 
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']

	else:
		optimizer = optim.Adam(frcnn.parameters(), lr=cfg.TRAIN.ADAM_LR)
		epoch = 0
		loss = 0
else:
	# ## When you are running for the first time.
	# with open(checkpoint_path, 'w') as f:
	# 	f.writelines('')
	optimizer = optim.Adam(frcnn.parameters(), lr=cfg.TRAIN.ADAM_LR)
	epoch = 0
	loss = 0

## Initializing RPN biases

loss_object = RPNLoss(cfg)

rpn_target = RPN_targets(cfg)
if cfg.USE_CUDA:
	frcnn = frcnn.cuda()
	loss_object = loss_object.cuda()
	# optimizer = optimizer.cuda()
	cfg.DTYPE.FLOAT = 'torch.cuda.FloatTensor'
	cfg.DTYPE.LONG = 'torch.cuda.LongTensor'


epochs = cfg.TRAIN.EPOCHS
## Learning rate scheduler
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_EPOCHS, gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)

frcnn.train()

while epoch <= epochs:
	epoch += 1
	image_number = 0
	running_loss = 0

	for images, labels in trainloader:
		
		# get ground truth in correct format
		image_number += 1
		if cfg.USE_CUDA:
			input_image = images.cuda()

		## If there are no ground truth objects in an image, we do this to not run into an error
		if len(labels) is 0:
			continue

		targets = process_coco_labels(labels)
		optimizer.zero_grad()
		prediction, out = frcnn.forward(input_image)
		# print(targets['boxes'])
		try:
			valid_anchors, valid_labels, _ = rpn_target.get_targets(input_image, out, targets)
		except:
			print("Inside exception!")
			continue
		target = {}
		target['gt_bbox'] = torch.unsqueeze(torch.from_numpy(valid_anchors),0)
		target['gt_anchor_label'] = torch.unsqueeze(torch.from_numpy(valid_labels).long(), 0) 
		valid_indices = np.where(valid_labels != -1)
		prediction['bbox_pred'] = prediction['bbox_pred'].type(cfg.DTYPE.FLOAT)
		prediction['bbox_uncertainty_pred'] = prediction['bbox_uncertainty_pred'].type(cfg.DTYPE.FLOAT)
		prediction['bbox_class'] = prediction['bbox_class'].type(cfg.DTYPE.FLOAT)
		target['gt_bbox'] = target['gt_bbox'].type(cfg.DTYPE.FLOAT)
		target['gt_anchor_label'] = target['gt_anchor_label'].type(cfg.DTYPE.LONG)
		loss = loss_object(prediction, target, valid_indices)

		if math.isnan(loss.item()):
			print("NaN detected.")
			continue
		# print(loss.item(), loss, loss.type(), targets)
		# print(loss_object.pos_anchors, loss_object.neg_anchors)
		
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		print("Classification loss is:", loss_object.class_loss.item(), " and regression loss is:", loss_object.reg_loss.item())
		print(f"Training loss: {loss.item()}", " epoch and image_number: ", epoch, image_number)

		### Save model and other things at every 10000 images.
		### TODO: Make this number a variable for config file

		if image_number%25000 == 0:
			### Save model!
			model_path = model_dir_path + str(image_number).zfill(10) +  str(epoch).zfill(5) + '.model'
			torch.save({
					'epoch': epoch,
					'model_state_dict': frcnn.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss,
					'cfg': cfg
					 }, model_path)

			with open(checkpoint_path, 'w') as f:
				f.writelines(model_path)		

	print(f"Running loss: {running_loss/len(trainloader)}")

	## For learing rate decay
	lr_scheduler.step()

	## Saving at the end of the epoch
	model_path = model_dir_path + "end_of_epoch_" + str(image_number).zfill(10) +  str(epoch).zfill(5) + '.model'
	torch.save({
			'epoch': epoch,
			'model_state_dict': frcnn.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': running_loss,
			'cfg': cfg
			 }, model_path)

	with open(checkpoint_path, 'w') as f:
		f.writelines(model_path)
