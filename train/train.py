"""
Training RPNs. 
"""

import torch
import sys
import numpy as np
import math
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
coco_dataset = dset.CocoDetection('/home/coco_dataset_new/train2017_modified/', '/home/coco_dataset_new/instances_train2017_modified.json', transform= tranform) 
trainloader = torch.utils.data.DataLoader(coco_dataset, batch_size=1, shuffle=True)

# Generate random input
# TODO: replace with actual image later,with vision tranforms(normalization)
# input_image, labels = iter(trainloader).next()
# targets = process_coco_labels(labels)
# print(targets)



frcnn = FRCNN(cfg)
for params in frcnn.backbone_obj.parameters():
	params.requires_grad = False

## Initialize RPN params

optimizer = optim.Adam(frcnn.parameters(), lr=1e-5)

model_dir_path = '/home/Denso_models/'

checkpoint_path = model_dir_path + 'checkpoint.txt'

if path.exists(checkpoint_path):
	with open(checkpoint_path, "r") as f: 
		model_path = f.readline().strip('\n')

	## Only load if such a model exists
	if path.exists(model_path):

		checkpoint = torch.load(model_path)
		frcnn.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']

	else:
		epoch = 0
		loss = 0
else:
	# ## When you are running for the first time.
	# with open(checkpoint_path, 'w') as f:
	# 	f.writelines('')
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


epochs = 50
frcnn.train()
while epoch <= epochs:
	epoch += 1
	running_loss = 0
	for images, labels in trainloader:
		break
		# get ground truth in correct format
		if cfg.USE_CUDA:
			input_image = images.cuda()

		## If there are no ground truth objects in an image, we do this to not run into an error
		if len(labels) is 0:
			continue

		targets = process_coco_labels(labels)
		# TODO: Training pass
		optimizer.zero_grad()
		prediction, out = frcnn.forward(input_image)
		# print(targets['boxes'])
		try:
			valid_anchors, valid_labels = rpn_target.get_targets(input_image, out, targets)
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
		print(f"Training loss: {running_loss/len(trainloader)}")


	### Save model!
	model_path = model_dir_path + str(epoch).zfill(5) + '.model'
	torch.save({
			'epoch': epoch,
			'model_state_dict': frcnn.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
			'cfg': cfg
			 }, model_path)

	with open(checkpoint_path, 'w') as f:
		f.writelines(model_path)

