import torch
import numpy as np
import time
from ..utils import utils
import os
import sys
from torch import optim
from torch.utils import tensorboard
from ..eval.detection_map import DetectionMAP
import matplotlib.pyplot as plt
from src.config import Cfg as cfg
from ..tracker.track import MultiObjTracker
from ..utils import Instances
from ..utils import Boxes



class Solver(object):
	"""docstring for Trainer"""
	def __init__(self, cfg, mode, args):
		super(Solver, self).__init__()

		self.device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
		print("Using the device for training: {} \n".format(device))
		self.setup_dirs(cfg, args.name)

		self.model = self.build_model(cfg)
		self.train_loader, self.val_loader = self.build_dataloader(cfg, mode)
		self.optimizer, self.lr_scheduler = self.build_optimizer(cfg)

		self.tb_writer = tensorboard.SummaryWriter(os.path.join(self.exp_dir,"tf_summary"))
		self.is_training = False
		self.epoch = 0

	def train_step(self, batch_sample):
		in_images = batch_sample['image'].to(self.device)
		target = [x.to(self.device) for x in batch_sample['target']]
		
		rpn_proposals, instances, proposal_losses, detector_losses = self.model(in_images, target, self.is_training)
		
		loss_dict = {}
		loss_dict.update(proposal_losses)
		loss_dict.update(detector_losses)
		
		loss = 0.0
		for k, v in loss_dict.items():
			loss += v
		loss_dict.update({'tot_loss':loss})

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss_dict

	def validation_step(self):
		val_loss = {}
		self.is_training = False
		with torch.no_grad():
			for idx, batch_sample in enumerate(val_loader):

				in_images = batch_sample['image'].to(self.device)
				target = [x.to(self.device) for x in batch_sample['target']]

				rpn_proposals, instances, proposal_losses, detector_losses = model(in_images, target, self.is_training)

				loss_dict = {}
				loss_dict.update(proposal_losses)
				loss_dict.update(detector_losses)

				loss = 0.0
				for k, v in loss_dict.items():
					loss += v
				loss_dict.update({'tot_loss':loss})

				for key, value in loss_dict.items():
					if len(val_loss)<len(loss_dict):
						val_loss[key] = []
					val_loss[key].append(loss_dict[key].item())

				# utils.tb_logger(in_images, tb_writer, rpn_proposals, instances, "Validation")

			for key, value in  val_loss.items():
				val_loss[key] = np.mean(val_loss[key])

		return val_loss
	
	def train(self, epochs, saving_freq):
		assert self.model.training()
		self.is_training = True
		while self.epoch <= epochs:

			running_loss = {}
			print("Epoch | Iteration | Loss ")

			for idx, batch_sample in enumerate(self.train_loader):
				loss_dict = self.train_step(batch_sample)
				
				if (idx)%10==0:
					with torch.no_grad():
						#----------- Logging and Printing ----------#
						print("{:<8d} {:<9d} {:<7.4f}".format(epoch, idx, loss_dict['tot_loss'].item()))
						for loss_name, value in loss_dict.items():
							self.tb_writer.add_scalar('Loss/'+loss_name, value.item(), epoch+0.01*idx)
						for key, value in loss_dict.items():
							if len(running_loss)<len(loss_dict):
								running_loss[key] = 0.0
							running_loss[key] = 0.9*running_loss[key] + 0.1*loss_dict[key].item()
						# utils.tb_logger(in_images, tb_writer, rpn_proposals, instances, "Training")
					#------------------------------------------------#
			val_loss = self.validation_step()
			
			for key in val_loss.keys():
				tb_writer.add_scalars('loss/'+key, {'validation': val_loss[key], 'train': running_loss[key]}, epoch)

			# print("Epoch ---- {} ".format(self.epoch))
			print("Epoch      Training   Validation")
			print("{} {:>13.4f}    {:0.4f}".format(epoch, running_loss['tot_loss'], val_loss['tot_loss']))

			## Decaying learning rate
			self.lr_scheduler.step()

			print("Epoch Complete: ", self.epoch)
			# # Saving at the end of the epoch
			if self.epoch % saving_freq == 0:
				self.save_checkpoint()

			self.epoch += 1

		self.tb_writer.close()

	def save_checkpoint(self):
		model_path = os.path.join(self.exp_dir,"models", "epoch_"+str(epoch).zfill(5)+ ".model")
		torch.save({
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'lr_scheduler' : self.lr_scheduler.state_dict(),
				 }, model_path)

	def build_model(self, cfg):
		print("--- Building the Model \n")
		# print(cfg.ROI_HEADS.SCORE_THRESH_TEST)
		model = FasterRCNN(cfg)
		model = model.to(self.device)

	def build_dataloader(self, cfg, mode):
		dataset_dir = {"detection": KittiDataset, "tracking": KittiMOTDataset}
		dataset = dataset_dir[cfg.DATASET.NAME]
		dataset_path = cfg.DATASET.PATH
		assert os.path.exists(self.dataset_path), "Dataset path doesn't exist."
		
		transform = utils.image_transform(cfg) # this is tranform to normalise/standardise the images
		batch_size = cfg.TRAIN.BATCH_SIZE

		print("--- Loading Dataset \n ")

        tracks = [str(i).zfill(4) for i in range(15)]
		train_dataset = dataset(dataset_path, tracks,transform = transform, cfg = cfg) #---- Dataloader
        tracks = [str(i).zfill(4) for i in range(15,21)]
		val_dataset = dataset(dataset_path, tracks, transform=transform, cfg=cfg)
		# dataset_len = len(dataset)
		## Split into train & validation
		# train_len = int(cfg.TRAIN.DATASET_DIVIDE*dataset_len)
		# val_len = dataset_len - train_len

		print("--- Data Loaded---")
		print("Number of Images in Train Dataset: {} \n".format(len(train_dataset)))
		print("Number of Images in Val Dataset: {} \n".format(len(val_dataset)))
		print("Number of Classes in Dataset: {} \n".format(cfg.INPUT.NUM_CLASSES))

		# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
		            shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)

		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
		            shuffle=False, collate_fn = kitti_collate_fn, drop_last=True)

		return train_loader, val_loader

	def build_optimizer(self, cfg):
		if cfg.TRAIN.OPTIM.lower() == 'adam':
		    optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
		elif cfg.TRAIN.OPTIM.lower() == 'sgd':
		    optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0.01)
		else:
		    raise ValueError('Optimizer must be one of \"sgd\" or \"adam\"')

		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.MILESTONES,
                		gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)

		return optimizer, lr_scheduler

	def setup_dirs(self, cfg, name):
		self.exp_dir = os.path.join(cfg.LOGS.BASE_PATH, name)
		if not os.path.exist(exp_dir):
			os.mkdir(experiment_dir)
	        os.mkdir(os.path.join(self.exp_dir+"models"))
	        os.mkdir(os.path.join(self.exp_dir+"results"))
	        os.mkdir(os.path.join(self.exp_dir+"tf_summary"))
	    else:
	    	"Directory exists, You may want to resume instead of starting fresh."
	
	def load_checkpoint(self, path):
		pass
	
