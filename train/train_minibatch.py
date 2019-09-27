"""
The purpose of this file is to train over a very small 
miniset of the dataset to see if the loss is converging 
or not. This may or may not be used once the training procedure 
is established
Training RPNs. 
"""


"""
How to run on MILA cluster?

python train_minibatch.py -dp "/network/tmp1/bhattdha/coco_dataset_new/train2017_modified/" -ap "/network/tmp1/bhattdha/coco_dataset_new/annotations_modified/instances_train2017_modified.json" -mp "/network/tmp1/bhattdha/Denso-miniset-test/"

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

minibatch_size = 10
### let's generate the dataset
tranform = image_transform(cfg)
coco_dataset = dset.CocoDetection(dset_path, ann_path, transform= tranform) 

## We are getting only a smaller dataset as we don't needs a full-fledged training
coco_part_tr = torch.utils.data.random_split(coco_dataset, [minibatch_size, len(coco_dataset)-minibatch_size])[0] ## Sampling a small minibatch
trainloader = torch.utils.data.DataLoader(coco_part_tr, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE)
print("Length of train set is: ", len(coco_part_tr), len(trainloader))

cfg.TRAIN.ADAM_LR=1e-3
cfg.TRAIN.FREEZE_BACKBONE = False
## The model
frcnn = FRCNN(cfg)
if cfg.TRAIN.FREEZE_BACKBONE:
	for params in frcnn.backbone_obj.parameters():
		params.requires_grad = False


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
		if cfg.USE_CUDA:
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

cfg.TRAIN.EPOCHS = 4000
epochs = cfg.TRAIN.EPOCHS
frcnn.train()

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)

while epoch <= epochs:
	epoch += 1
	image_number = 0
	running_loss = 0

	fake_batch = 0
	for images, labels in trainloader:
		
		# fake_batch += 1
		# ## Start new epoch 
		# if image_number > minibatch_size: 
		# 	break

		# get ground truth in correct format
		image_number += 1
		if cfg.USE_CUDA:
			input_image = images.cuda()

		## If there are no ground truth objects in an image, we do this to not run into an error
		if len(labels) is 0:
			continue

		targets = process_coco_labels(labels)
		# TODO: Training pass
		# optimizer.zero_grad()
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
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		# print(loss.item(), loss, loss.type(), targets)
		# print(loss_object.pos_anchors, loss_object.neg_anchors)
		# print(f"Training loss: {loss.item()}", " epoch and image_number: ", epoch, image_number)
		# if fake_batch == 13:
		# 	print("inside backward")
		# 	loss.backward()
		# 	optimizer.step()
		# 	optimizer.zero_grad()
		# 	# fake_batch = 0


		running_loss += loss.item()
		

		# if fake_batch == 13:
		# 	fake_batch = 0
		# 	loss = 0

		

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

	lr_scheduler.step()
	print(f"Running loss: {running_loss}")

	
	## Saving at the end of the 50 epochs
	if epoch % 1000 == 0:
		model_path = model_dir_path + "end_of_epoch_" + str(image_number).zfill(10) +  str(epoch).zfill(5) + '.model'
		torch.save({
				'epoch': epoch,
				'model_state_dict': frcnn.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': running_loss/len(trainloader),
					'cfg': cfg
				 }, model_path)

		with open(checkpoint_path, 'w') as f:
			f.writelines(model_path)
