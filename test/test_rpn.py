"""
Testing RPNs. 
"""


"""
How to run on MILA cluster?

python test_rpn.py -dp "/network/tmp1/bhattdha/coco_dataset_new/train2017_modified/" -ap "/network/tmp1/bhattdha/coco_dataset_new/annotations_modified/instances_train2017_modified.json" -mp "/network/tmp1/bhattdha/Denso-miniset-test/end_of_epoch_000000001004000.model"

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
from src.datasets import CocoDetection_modified
from torchvision import datasets as dset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


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
# if not path.exists(model_dir_path):
# 	os.mkdir(model_dir_path)

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
coco_dataset = CocoDetection_modified(dset_path, ann_path, transform= tranform) 

## We are getting only a smaller dataset as we don't needs a full-fledged training
coco_part_tr = torch.utils.data.random_split(coco_dataset, [minibatch_size, len(coco_dataset)-minibatch_size])[0] ## Sampling a small minibatch
trainloader = torch.utils.data.DataLoader(coco_part_tr, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE)
print("Length of train set is: ", len(coco_part_tr), len(trainloader))

cfg.TRAIN.ADAM_LR=1e-3
cfg.TRAIN.FREEZE_BACKBONE = False
## The model
frcnn = FRCNN(cfg)
model_path = args["modelpath"]
checkpoint = torch.load(model_path)
frcnn.load_state_dict(checkpoint['model_state_dict'])
loss_object = RPNLoss(cfg)

rpn_target = RPN_targets(cfg)
if cfg.USE_CUDA:
	frcnn = frcnn.cuda()
	loss_object = loss_object.cuda()
	# optimizer = optimizer.cuda()
	cfg.DTYPE.FLOAT = 'torch.cuda.FloatTensor'
	cfg.DTYPE.LONG = 'torch.cuda.LongTensor'

frcnn.eval()

def get_actual_coords(prediction, anchors):
	prediction = prediction.detach().cpu().numpy()
	prediction = prediction.reshape([prediction.shape[1], prediction.shape[2]])


	y_c = prediction[:,0]*(anchors[:,2] - anchors[:,0]) + anchors[:,0] + 0.5*(anchors[:,2] - anchors[:,0])
	x_c = prediction[:,1]*(anchors[:,3] - anchors[:,1]) + anchors[:,1] + 0.5*(anchors[:,3] - anchors[:,1])
	h = np.exp(prediction[:,2])*(anchors[:,2] - anchors[:,0])
	w = np.exp(prediction[:,3])*(anchors[:,3] - anchors[:,1])

	x1 = x_c - w/2.0
	y1 = y_c - h/2.0

	bbox_locs = np.vstack((x1, y1, w, h)).transpose() ## Final locations of the anchors

	# print(type(prediction), prediction.shape, anchors.shape)
	return bbox_locs

image_number = 0

for images, labels, img_name in trainloader:
	
	image_number += 1

	# if image_number < 7:
	# 	continue

	if cfg.USE_CUDA:
		input_image = images.cuda()

	print(img_name[0])

	## If there are no ground truth objects in an image, we do this to not run into an error
	if len(labels) is 0:
		continue

	targets = process_coco_labels(labels)
	# TODO: Training pass
	# optimizer.zero_grad()
	prediction, out = frcnn.forward(input_image)
	# print(targets['boxes'])
	try:
		valid_anchors, valid_labels, orig_anchors = rpn_target.get_targets(input_image, out, targets)
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
	prediction['bbox_class'] = torch.nn.functional.softmax(prediction['bbox_class'].type(cfg.DTYPE.FLOAT), dim=2)
	target['gt_bbox'] = target['gt_bbox'].type(cfg.DTYPE.FLOAT)
	target['gt_anchor_label'] = target['gt_anchor_label'].type(cfg.DTYPE.LONG)

	## To avoid overflow?
	for i in np.arange(prediction['bbox_class'].size()[1]):
		if prediction['bbox_class'][0,i,:][1].item() < 0.8:
			prediction['bbox_pred'][0,i,:] = 0



	bbox_locs = get_actual_coords(prediction['bbox_pred'], orig_anchors)

	im = np.array(Image.open(img_name[0]), dtype=np.uint8)

	# Create figure and axes
	fig,ax = plt.subplots(1)

	# Display the image
	ax.imshow(im)

	for i in np.arange(prediction['bbox_class'].size()[1]):
		if prediction['bbox_class'][0,i,:][1].item() > 0.999:
			rect = patches.Rectangle((bbox_locs[i][0],bbox_locs[i][1]),bbox_locs[i][2],bbox_locs[i][3],linewidth=1,edgecolor='r',facecolor='none')		
			ax.add_patch(rect)


	# Create a Rectangle patch
	# rect = patches.Rectangle((50,100),100,300,linewidth=1,edgecolor='r',facecolor='none')

	# Add the patch to the Axes


	# plt.show()
	fig.savefig('/home/dhaivat1729/temp' + str(image_number) + '.png', dpi=fig.dpi)
	# break
	# print("Bounding boxes are:" prediction['bbox_pred'])
	# print("bbox_class")


			# print("Box and coords are: ", prediction['bbox_class'][0,i,:], prediction['bbox_pred'][0,i,:])



## plot and shizz







