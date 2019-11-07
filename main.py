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
## Inserting path of src directory
# sys.path.insert(1, '../')

from src.architecture import FasterRCNN
from src.config import Cfg as cfg # Configuration file
from src.datasets import process_kitti_labels
from src.datasets import kitti_collate_fn
from src.datasets import KittiDataset # Dataloader
from src.utils import utils, Boxes
# from src.pytorch_nms import nms as NMS


from torchvision import datasets as dset
from torchvision import transforms as T
import torchvision
from torch.utils import tensorboard


#----- Initial paths setup and loading config values ------ #

print("\n--- Setting up the training/testing \n")

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")
ap.add_argument("-mode", "--mode",required = True, choices=['train', 'test'])

args = vars(ap.parse_args())

dataset_path = cfg.PATH.DATASET
experiment_dir = cfg.PATH.LOGS + "/" + args["experiment_comment"]

results_dir = experiment_dir+"/results"
graph_dir = experiment_dir+"/tf_summary"
model_save_dir = experiment_dir+"/models"

if not path.exists(dataset_path):
	print("Dataset path doesn't exist")
if not path.exists(experiment_dir):
	os.mkdir(experiment_dir)
	os.mkdir(results_dir)
	os.mkdir(model_save_dir)
	os.mkdir(graph_dir)

device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")

mode = args['mode']

if mode=='train':
	is_training = True
else:
	is_training = False

torch.manual_seed(cfg.RANDOMIZATION.SEED)
np.random.seed(cfg.RANDOMIZATION.SEED)
tb_writer = tensorboard.SummaryWriter(graph_dir)

#-----------------------------------------------#


#---------Modelling and Trainer Building------#
print("--- Building the Model \n")

model = FasterRCNN(cfg)
model = model.to(device)
model.eval()

if cfg.TRAIN.OPTIM.lower() == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
elif cfg.TRAIN.OPTIM.lower() == 'sgd':
	optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0.01)
else:
	raise ValueError('Optimizer must be one of \"sgd\" or \"adam\"')

#-------- Dataset loading and manipulation-------#

transform, inv_transform = utils.image_transform(cfg) # this is tranform to normalise/standardise the images

if mode=="train":
	print("--- Loading Training Dataset \n ")
	kitti_dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg) #---- Dataloader
	dataset_len = len(kitti_dataset)
	## Split into train & validation
	train_len = int(cfg.TRAIN.DATASET_DIVIDE*dataset_len)
	val_len = dataset_len - train_len

	print("--- Data Loaded--- Number of Images in Dataset: {} \n".format(dataset_len))

	kitti_train_dataset, kitti_val_dataset = torch.utils.data.random_split(kitti_dataset, [train_len, val_len])

	kitti_train_loader = torch.utils.data.DataLoader(kitti_train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
		shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)
	kitti_val_loader = torch.utils.data.DataLoader(kitti_val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
		shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)

else:
	print("---Loading Testing Dataset")
	kitti_test_dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg) #---- Dataloader
	kitti_test_loader = torch.utils.data.DataLoader(kitti_test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn)
	print("---Data Loaded---- Number of Images in Dataset: ", len(kitti_dataset))

#----------------------------------------------#

#---------Training Cycle-----------#
print("Starting the training in 3.   2.   1.   Go")

epochs = cfg.TRAIN.EPOCHS
epoch = 0
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)

while epoch <= epochs:

	epoch += 1
	running_loss = 0.0
	running_loss_classify = 0.
	running_loss_regress = 0.

	for idx, batch_sample in enumerate(kitti_train_loader):

		# print(batch_sample['image'].device, batch_sample['boxes'][0].device, batch_sample['class_labels'][0].device)

		in_images = batch_sample['image'].to(device)
		gt_boxes = [x.to(device) for x in batch_sample['boxes']]
		class_labels = [x.to(device) for x in batch_sample['class_labels']]

		proposals, rpn_losses = model(in_images, gt_boxes, is_training)
			
		class_loss = rpn_losses['loss_rpn_cls']
		regress_loss  = rpn_losses['loss_rpn_loc']

		loss = class_loss + regress_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (idx)%10==0:
			
			#----------- Logging and Printing ----------#
			print("Epoch | Iteration | Loss | Bbox | Class")
			print("{:<8d} {:<9d} {:<7.4f} {:<7.4f} {:<8.4f}".format(epoch, idx, loss.item(), regress_loss.item(), class_loss.item()))

			# print("Epoch | Iteration | Loss | Bbox | Class | Error")
			# print("{:<8d} {:<9d} {:<7.4f} {:<7.4f} {:<8.4f}{:<8.4f}".format(epoch, idx, loss.item(), loss_regress_bbox.item(), loss_classify.item(), loss_error_bbox.item()))
		
			tb_writer.add_scalar('Loss/Total_Loss', loss.item(), epoch+0.01*idx)
			tb_writer.add_scalar('Loss/Classification', class_loss.item(), epoch+0.01*idx)
			tb_writer.add_scalar('Loss/Regression', regress_loss.item(), epoch+0.01*idx)
			# tb_writer.add_scalar('Loss/Error', loss_error_bbox.item(), epoch+itr_num)
			
			running_loss = 0.9*running_loss + 0.1*loss
			running_loss_classify = 0.9*running_loss_classify + 0.1*class_loss
			running_loss_regress = 0.9*running_loss_regress + 0.1*regress_loss
			# running_loss_error = 0.9*running_loss_error + 0.1*batch_loss_error

			with torch.no_grad():
				img_path = batch_sample["image_path"][2]
				bbox_locs = proposals[2].proposal_boxes[:50].tensor

				image = np.array(Image.open(img_path), dtype='uint8')
				
				# pos_img, _ = utils.draw_bbox(image,utils.xy_to_wh(pos_bbox))
				# predict_img, _ = utils.draw_bbox(image, bbox_locs.tensor)				
				image_grid, _ = utils.draw_bbox(image, bbox_locs)				

				# image_grid = np.concatenate([pos_img,predict_img], axis = 1)
				# print(image_grid.shape)
				# image_grid = torchvision.utils.make_grid(torch.stack(image_grid), 1)

				tb_writer.add_image('Training Image', image_grid, dataformats='HWC')

			#------------------------------------------------#

	val_loss = []
	val_loss_classify = []
	val_loss_regress = []
	# val_loss_error = []

	with torch.no_grad():
		for idx, batch_sample in enumerate(kitti_val_loader):
			
			in_images = batch_sample['image'].to(device)
			gt_boxes = [x.to(device) for x in batch_sample['boxes']]
			class_labels = [x.to(device) for x in batch_sample['class_labels']]

			proposals, rpn_losses = model(in_images, gt_boxes,is_training=True)

			class_loss = rpn_losses['loss_rpn_cls']
			regress_loss  = rpn_losses['loss_rpn_loc']

			loss = class_loss + regress_loss

			val_loss.append(loss.item())
			val_loss_classify.append(class_loss.item())
			val_loss_regress.append(regress_loss.item())
			# val_loss_error.append(loss_error_bbox.item())


			img_path = batch_sample["image_path"][2]
			bbox_locs = proposals[2].proposal_boxes[:50].tensor

			image = np.array(Image.open(img_path), dtype='uint8')
			
			# pos_img, _ = utils.draw_bbox(image,utils.xy_to_wh(pos_bbox))
			# predict_img, _ = utils.draw_bbox(image, bbox_locs.tensor)				
			image_grid, _ = utils.draw_bbox(image, bbox_locs)				

			# image_grid = np.concatenate([pos_img,predict_img], axis = 1)
			# print(image_grid.shape)
			# image_grid = torchvision.utils.make_grid(torch.stack(image_grid), 1)
			tb_writer.add_image('Validation Image', image_grid, dataformats='HWC')

		val_loss = np.mean(val_loss)
		val_loss_classify = np.mean(val_loss_classify)
		val_loss_regress = np.mean(val_loss_regress)
		# val_loss_error = np.mean(val_loss_error)

		tb_writer.add_scalars('loss/classification', {'validation': val_loss_classify, 'train': running_loss_classify.item()}, epoch)
		tb_writer.add_scalars('loss/regression', {'validation': val_loss_regress, 'train': running_loss_regress.item()}, epoch)
		# tb_writer.add_scalars('loss/error', {'validation': val_loss_error, 'train': running_loss_error.item()}, epoch)


		print("Epoch ---- {} ".format(epoch))
		print("           Training   Validation")
		print("Loss: {:>13.4f}    {:0.4f}".format(running_loss.item(), val_loss))
		print("Training loss: {} classification: {} Regression: {}".format(running_loss.item(), 
			running_loss_classify.item(), running_loss_regress.item()))
		print("Validation loss: ", val_loss, "Classification loss: ", val_loss_classify, "Regression Loss: ", val_loss_regress)
		
	## Decaying learning rate
	lr_scheduler.step()

	print("Epoch Complete: ", epoch)
	# # Saving at the end of the epoch
	if epoch % cfg.TRAIN.SAVE_MODEL_EPOCHS == 0:
		model_path = model_save_dir + "/epoch_" +  str(epoch).zfill(5) + '.model'
		torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': running_loss,
				'cfg': cfg
				 }, model_path)

		# with open(checkpoint_path, 'w') as f:
		# 	f.writelines(model_path)

tb_writer.close()
# file.close()
