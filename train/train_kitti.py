"""
Training RPNs. 
"""


"""
How to run on MILA cluster?

python train_kitti.py -name "bla bla bla" 
"""

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
sys.path.insert(1, '../')

from src.architecture import FRCNN
from src.config import Cfg as cfg # Configuration file
from src.RPN import anchor_generator, RPN_targets
from src.preprocess import image_transform ## It's a function, not a class.  
from src.datasets import process_kitti_labels
from src.datasets import kitti_collate_fn
from src.datasets import KittiDataset # Dataloader
from src.loss import RPNErrorLoss
from src.utils import utils
from src.NMS import nms as NMS

from torchvision import datasets as dset
from torchvision import transforms as T
import torchvision
from torch.utils import tensorboard




#----- Initial paths setup and loading config values ------ #

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")

args = vars(ap.parse_args())

dset_path = cfg.PATH.DATASET
experiment_dir = cfg.PATH.LOGS + "/" + args["experiment_comment"]

results_dir = experiment_dir+"/results"
graph_dir = experiment_dir+"/tf_summary"
model_save_dir = experiment_dir+"/models"

if not path.exists(dset_path):
	print("Dataset path doesn't exist")
if not path.exists(experiment_dir):
	os.mkdir(experiment_dir)
	os.mkdir(experiment_dir+"/results")
	os.mkdir(experiment_dir+"/models")
	os.mkdir(experiment_dir+"/tf_summary")


print("reached here!!!!")
file = open(experiment_dir+"/train_log.txt", 'w')

# Setting the seeds
torch.manual_seed(cfg.RANDOMIZATION.SEED)
np.random.seed(cfg.RANDOMIZATION.SEED)

## setting default variable types
torch.set_default_tensor_type('torch.FloatTensor') 
torch.set_default_dtype(torch.float32)

### use cuda only if it's available and permitted

if torch.cuda.is_available() and not cfg.NO_GPU:
	cfg.USE_CUDA = True

#-----------------------------------------------#


#-------- Dataset loading and manipulation-------#

transform, inv_transform = image_transform(cfg) # this is tranform to normalise/standardise the images

kitti_dataset = KittiDataset(dset_path, transform = transform, cfg = cfg) #---- Dataloader
print("Number of Images in Dataset: ", len(kitti_dataset))

## Split into train & validation

if cfg.TRAIN.TRAIN_LENGTH and cfg.TRAIN.VAL_LENGTH:
	train_len = cfg.TRAIN.TRAIN_LENGTH
	val_len = cfg.TRAIN.VAL_LENGTH
else:
	train_len = int(cfg.TRAIN.DATASET_DIVIDE*len(kitti_dataset))
	val_len = len(kitti_dataset) - train_len

kitti_train_dataset, kitti_val_dataset = torch.utils.data.random_split(kitti_dataset, [train_len, val_len])

## Dataloader for training
kitti_train_loader = torch.utils.data.DataLoader(kitti_train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn)
kitti_val_loader = torch.utils.data.DataLoader(kitti_val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn)

#----------------------------------------------#

tb_writer = tensorboard.SummaryWriter(graph_dir)

#--------- Define the model ---------------#

frcnn = FRCNN(cfg)
if cfg.TRAIN.FREEZE_BACKBONE:
	for params in frcnn.backbone_obj.parameters():
		params.requires_grad = False

## Initialize RPN params
if cfg.TRAIN.OPTIM.lower() == 'adam':
	optimizer = optim.Adam(frcnn.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
elif cfg.TRAIN.OPTIM.lower() == 'sgd':
	optimizer = optim.SGD(frcnn.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0.01)
else:
	raise ValueError('Optimizer must be one of \"sgd\" or \"adam\"')

loss_object = RPNErrorLoss(cfg)

rpn_target = RPN_targets(cfg)
if cfg.USE_CUDA:
	frcnn = frcnn.cuda()
	loss_object = loss_object.cuda()
	# optimizer = optimizer.cuda()
	cfg.DTYPE.FLOAT = 'torch.cuda.FloatTensor'
	cfg.DTYPE.LONG = 'torch.cuda.LongTensor'
#-----------------------------------------#


#--------- Training Procedure -------------#

#------- Loading previous point or running new----------#

checkpoint_path = experiment_dir + '/checkpoint.txt'
print(checkpoint_path)
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

		## For multistep lr schedular
		new_milestones = list(np.array((list(cfg.TRAIN.MILESTONES))) - epoch)

		## If we have already achieved some milestones, there will be negative numbers for them
		## Let's get rid of them
		for i in range(len(new_milestones)):
			if new_milestones[0] <= 0:
				new_milestones.pop(0)

		cfg.TRAIN.MILESTONES = tuple(new_milestones)
		# print(cfg.TRAIN.MILESTONES)
		# sys.exit()
		loss = checkpoint['loss']

	else:
		optimizer = optim.Adam(frcnn.parameters(), lr=cfg.TRAIN.ADAM_LR, weight_decay=0.0005)
		epoch = 0
		loss = 0
else:
	# ## When you are running for the first time.
	epoch = 0
	loss = 0

#-----------------------------------------#


epochs = cfg.TRAIN.EPOCHS

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)


tb_writer.add_graph(frcnn, torch.normal(mean=1e-2, std=1.0, size=(1, 3, 345, 1242)).cuda())  #Give some random input here as the same size as your input to the model

frcnn.eval()
while epoch <= epochs:
	epoch += 1
	image_number = 0
	running_loss = 0.0
	running_loss_classify = 0.
	running_loss_regress = 0.
	running_loss_error = 0.0
	batch_loss = 0.0

	for idx, (image, labels, paths) in enumerate(kitti_train_loader):


		input_image = image
		if cfg.USE_CUDA:
			input_image = input_image.cuda()

		## If there are no ground truth objects in an image, we do this to not run into an error
		if len(labels) is 0:
			continue

		targets = process_kitti_labels(cfg, labels)
		# optimizer.zero_grad()

		prediction, feat_map = frcnn.forward(input_image)
		# print(feat_map.shape)

		bboxes = prediction[0]
		class_probs = prediction[1]
		uncertainties = prediction[2]

		try:
			valid_anchors, valid_labels, orig_anchors = rpn_target.get_targets(input_image, feat_map, targets)
			# print( np.sum( valid_labels == 1) , np.sum( valid_labels == 0))
		except:
			print("Inside exception!")
			continue
		
		image_number += 1
		target = {}
		target['gt_bbox'] = torch.unsqueeze(torch.from_numpy(valid_anchors),0)
		target['gt_anchor_label'] = torch.unsqueeze(torch.from_numpy(valid_labels).long(), 0) 
		valid_indices = np.where(valid_labels != -1)
		bboxes = bboxes.type(cfg.DTYPE.FLOAT)
		uncertainties = uncertainties.type(cfg.DTYPE.FLOAT)
		class_probs = class_probs.type(cfg.DTYPE.FLOAT)
		target['gt_bbox'] = target['gt_bbox'].type(cfg.DTYPE.FLOAT)
		target['gt_anchor_label'] = target['gt_anchor_label'].type(cfg.DTYPE.LONG)

		loss_classify, loss_regress_bbox, loss_error_bbox = loss_object(prediction, target, valid_indices)

		loss = loss_error_bbox + loss_classify + loss_regress_bbox
		
		batch_loss += loss 
		batch_loss_classify += loss_classify
		batch_loss_bbox += loss_regress_bbox
		batch_loss_error += loss_error_bbox

		itr_num = idx/cfg.TRAIN.FAKE_BATCHSIZE

		if (idx+1)%cfg.TRAIN.FAKE_BATCHSIZE==0:
			batch_loss = batch_loss/cfg.TRAIN.FAKE_BATCHSIZE 
			batch_loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			
			#------------ Logging and Printing ----------#
			print("Epoch | Iteration | Loss | Bbox | Class | Error")
			print("{:8>d} {:9>d} {:7>0.4f} {:7>0.4f} {:8>0.4f} {:8>0.4f}".format(epoch, idx, loss.item(), loss_regress_bbox.item(), loss_classify.item(), loss_error_bbox.item()))
			
			tb_writer.add_scalar('Loss/Classification', loss_classify.item(), epoch+itr_num)
			tb_writer.add_scalar('Loss/Regression', loss_regress_bbox.item(), epoch+itr_num)
			tb_writer.add_scalar('Loss/Error', loss_error_bbox.item(), epoch+itr_num)

			batch_loss_classify = batch_loss_classify/cfg.TRAIN.FAKE_BATCHSIZE
			batch_loss_bbox = batch_loss_bbox/cfg.TRAIN.FAKE_BATCHSIZE
			batch_loss_error = batch_loss_error/cfg.TRAIN.FAKE_BATCHSIZE
			running_loss = 0.9*running_loss + 0.1*batch_loss
			running_loss_classify = 0.9*running_loss_classify + 0.1*batch_loss_classify
			running_loss_regress = 0.9*running_loss_regress + 0.1*batch_loss_bbox
			running_loss_error = 0.9*running_loss_error + 0.1*batch_loss_error
	
			#------------------------------------------------#

	rnd_indxs = np.random.randint(0, val_len-1, 10)

	val_loss = []
	val_loss_classify = []
	val_loss_regress = []
	val_loss_error = []

	with torch.no_grad():
		for idx, (image, labels, paths) in enumerate(kitti_val_loader):
			input_image = image

			if cfg.USE_CUDA:
				input_image = input_image.cuda()

			## If there are no ground truth objects in an image, we do this to not run into an error
			if len(labels) is 0:
				continue

			targets = process_kitti_labels(cfg, labels)
			
			prediction, feat_map = frcnn.forward(input_image)

			bboxes = prediction[0]
			class_probs = prediction[1]
			uncertainties = prediction[2]

			#why is here inside exception?
			try:
				valid_anchors, valid_labels, orig_anchors = rpn_target.get_targets(input_image, feat_map, targets)
			except:
				print("Inside exception!")
				continue
			
			image_number += 1
			target = {}
			target['gt_bbox'] = torch.unsqueeze(torch.from_numpy(valid_anchors),0)
			target['gt_anchor_label'] = torch.unsqueeze(torch.from_numpy(valid_labels).long(), 0) 
			valid_indices = np.where(valid_labels != -1)
			bboxes = bboxes.type(cfg.DTYPE.FLOAT)
			uncertainties = uncertainties.type(cfg.DTYPE.FLOAT)
			class_probs = class_probs.type(cfg.DTYPE.FLOAT)
			class_probs = torch.nn.functional.softmax(class_probs.type(cfg.DTYPE.FLOAT), dim=2)
			target['gt_bbox'] = target['gt_bbox'].type(cfg.DTYPE.FLOAT)
			target['gt_anchor_label'] = target['gt_anchor_label'].type(cfg.DTYPE.LONG)

			loss_classify, loss_regress_bbox, loss_error_bbox = loss_object(prediction, target, valid_indices)
			loss = loss_error_bbox + cfg.TRAIN.CLASS_LOSS_SCALE*loss_classify + cfg.TRAIN.SMOOTHL1LOSS_SCALE*loss_regress_bbox

			# print("Val loss - bbox: ", loss_regress_bbox.item(), " classification: ", loss_classify.item(), " Error:", loss_error_bbox.item())
			val_loss.append(loss.item())
			val_loss_classify.append(loss_classify.item())
			val_loss_regress.append(loss_regress_bbox.item())
			val_loss_error.append(loss_error_bbox.item())

			target['bbox_pred'] = target['gt_bbox']
			bbox_locs = utils.get_actual_coords(prediction, orig_anchors)
			# pos_bbox= utils.get_actual_coords(target, orig_anchors)	
			pos_bbox = orig_anchors[valid_labels==1]


			if idx in rnd_indxs:
				if cfg.NMS.USE_NMS==True:
					nms = NMS(cfg.NMS_THRES)
					index_to_keep = nms.apply_nms(bbox_locs, class_probs)
					index_to_keep = index_to_keep.numpy()
				else:
					index_to_keep = np.arrange(len(bbox_locs))

				final_indexes = []

				for box_idx in index_to_keep:
					if class_probs[0,box_idx,:][1].item() > 0.60:
						final_indexes.append(box_idx)

					
				print("Num detected boxes {}, Num of positive boxes {}".format(len(final_indexes), len(pos_bbox)))
				
				bbox_locs = bbox_locs[final_indexes]
				print(bbox_locs.shape, bbox_locs[:,4])	

				# print(image)
				# image = inv_transform(image)
				image = np.array(Image.open(paths[0]), dtype='uint8')
				
				# input_image = image
				pos_img, _ = utils.draw_bbox(image,utils.xy_to_wh(pos_bbox))
				predict_img, _ = utils.draw_bbox(image, bbox_locs, show_text=True)				

				# print(input_image.shape, pos_img.shape, predict_img.shape)
				image_grid = np.concatenate([pos_img,predict_img], axis = 1)
				# print(image_grid.shape)
				# image_grid = torchvision.utils.make_grid(torch.stack(image_grid), 1)

				tb_writer.add_image('Image', image_grid, epoch, dataformats='HWC')

		val_loss = np.mean(val_loss)
		val_loss_classify = np.mean(val_loss_classify)
		val_loss_regress = np.mean(val_loss_regress)
		val_loss_error = np.mean(val_loss_error)

		tb_writer.add_scalars('loss/classification', {'validation': val_loss_classify, 'train': running_loss_classify.item()}, epoch)
		tb_writer.add_scalars('loss/regression', {'validation': val_loss_regress, 'train': running_loss_regress.item()}, epoch)
		tb_writer.add_scalars('loss/error', {'validation': val_loss_error, 'train': running_loss_error.item()}, epoch)


		print("Epoch ------------- {} ".format(epoch))
		print("Training loss: {} classification: {} Regression: {} Error: {}".format(running_loss.item(), 
			running_loss_classify.item(), running_loss_regress.item(), running_loss_error.item()))
		print("Validation loss: ", val_loss, "Classification loss: ", val_loss_classify, "Regression Loss: ", val_loss_regress, 
			"Error Loss: ", val_loss_error)
		
	## Decaying learning rate
	lr_scheduler.step()

	print("Epoch Complete: ", epoch)
	# # Saving at the end of the epoch
	if epoch % cfg.TRAIN.SAVE_MODEL_EPOCHS == 0:
		model_path = model_save_dir + "/end_of_epoch_" + str(image_number).zfill(10) +  str(epoch).zfill(5) + '.model'
		torch.save({
				'epoch': epoch,
				'model_state_dict': frcnn.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': running_loss,
				'cfg': cfg
				 }, model_path)

		with open(checkpoint_path, 'w') as f:
			f.writelines(model_path)

tb_writer.close()
file.close()
