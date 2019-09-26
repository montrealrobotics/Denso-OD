"""
Testing RPNs. 
"""


"""
How to run on MILA cluster?

python test.py -dp "/network/tmp1/bhattdha/coco_dataset_new/val2017_modified/" -ap "/network/tmp1/bhattdha/coco_dataset_new/annotations_modified/val_train2017_modified.json" -mp "/home/Denso_models/000005000000013.model"

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
from src.NMS import nms_class

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--datasetpath", required = True, help="give dataset path")
ap.add_argument("-ap", "--annotationpath", required = True, help="give annotation file path")
ap.add_argument("-mp", "--modelpath", required = True, help="give model path")

args = vars(ap.parse_args())
dset_path = args["datasetpath"]
ann_path = args["annotationpath"]
model_path = args["modelpath"]

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

### let's generate the dataset
tranform = image_transform(cfg)
# coco_dataset = dset.CocoDetection(dset_path, ann_path, transform= tranform) 
# trainloader = torch.utils.data.DataLoader(coco_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE)





frcnn = FRCNN(cfg)
checkpoint = torch.load(model_path)
frcnn.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded successfully.")

# loss_object = RPNLoss(cfg)

# rpn_target = RPN_targets(cfg)

if cfg.USE_CUDA:
	frcnn = frcnn.cuda()
	# loss_object = loss_object.cuda()
	# optimizer = optimizer.cuda()
	cfg.DTYPE.FLOAT = 'torch.cuda.FloatTensor'
	cfg.DTYPE.LONG = 'torch.cuda.LongTensor'



frcnn.eval()

img = mpimg.imread('000000111179.jpg')		## Gives RGB image of dimension H x W x C with inten values between 0-255
print(img.shape)
tranform = image_transform(cfg)
# print(list(cfg.INPUT.MEAN))
input_image = tranform(img)
input_image = torch.unsqueeze(input_image, dim=0)
print(input_image.shape)
if cfg.USE_CUDA:
	input_image = input_image.cuda()


## let's do the forward pass!!! :D :D :D 
prediction, out = frcnn.forward(input_image)
print("forward pass successful.")

nms_object = nms_class(nms_thres = 0.6)
prediction['bbox_pred'] = prediction['bbox_pred'].type(cfg.DTYPE.FLOAT)
prediction['bbox_uncertainty_pred'] = prediction['bbox_uncertainty_pred'].type(cfg.DTYPE.FLOAT)
print(prediction['bbox_class'])
prediction['bbox_class'] = torch.nn.functional.softmax(prediction['bbox_class'].type(cfg.DTYPE.FLOAT), dim=2)
print("Shapes are", prediction['bbox_pred'].shape, prediction['bbox_uncertainty_pred'].shape, prediction['bbox_class'].shape)

for i in np.arange(prediction['bbox_class'].size()[1]):
	if prediction['bbox_class'][0,i,0] == 0.5:
		print(prediction['bbox_uncertainty_pred'][0,i,:])

# prediction['bbox_uncertainty_pred'].size(), prediction['bbox_class'])

# 		valid_indices = np.where(valid_labels != -1)

# 		target['gt_bbox'] = target['gt_bbox'].type(cfg.DTYPE.FLOAT)
# 		target['gt_anchor_label'] = target['gt_anchor_label'].type(cfg.DTYPE.LONG)
# 		loss = loss_object(prediction, target, valid_indices)

# 		if math.isnan(loss.item()):
# 			print("NaN detected.")
# 			continue
# 		# print(loss.item(), loss, loss.type(), targets)
# 		# print(loss_object.pos_anchors, loss_object.neg_anchors)
		
# 		loss.backward()
# 		optimizer.step()
# 		running_loss += loss.item()
# 		print("Classification loss is:", loss_object.class_loss.item(), " and regression loss is:", loss_object.reg_loss.item())
# 		print(f"Training loss: {loss.item()}", " epoch and image_number: ", epoch, image_number)

# 		### Save model and other things at every 10000 images.
# 		### TODO: Make this number a variable for config file

# 		if image_number%25000 == 0:
# 			### Save model!
# 			model_path = model_dir_path + str(image_number).zfill(10) +  str(epoch).zfill(5) + '.model'
# 			torch.save({
# 					'epoch': epoch,
# 					'model_state_dict': frcnn.state_dict(),
# 					'optimizer_state_dict': optimizer.state_dict(),
# 					'loss': loss,
# 					'cfg': cfg
# 					 }, model_path)

# 			with open(checkpoint_path, 'w') as f:
# 				f.writelines(model_path)		

# 	print(f"Running loss: {running_loss/len(trainloader)}")

# 	## For learing rate decay
# 	lr_scheduler.step()

# 	## Saving at the end of the epoch
# 	model_path = model_dir_path + "end_of_epoch_" + str(image_number).zfill(10) +  str(epoch).zfill(5) + '.model'
# 	torch.save({
# 			'epoch': epoch,
# 			'model_state_dict': frcnn.state_dict(),
# 			'optimizer_state_dict': optimizer.state_dict(),
# 			'loss': running_loss,
# 			'cfg': cfg
# 			 }, model_path)

# 	with open(checkpoint_path, 'w') as f:
# 		f.writelines(model_path)
