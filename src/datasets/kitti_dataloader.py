import torch
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset #Default dataloader class for Pytorch
import os
import glob
from ..utils import Boxes, Instances
import time

#Gives Memory error this way
class KittiDatasetList(Dataset):
	"""Kitti Dataset Reader."""

	def __init__(self, root_dir,transform=None, cfg = None):
		"""
		Args:
			root_dir (string): Path to the dataset.

			transform (callable, optional): Optional transform to be applied
				on a sample.

			cfg: config file
		"""

		self.cfg = cfg
		self.root_dir = root_dir
		self.transform = transform
		
		## has all the annotations 
		'''
		It's a dictionary, with keys() as absolute image_paths.
		Each key leads to a list of annotations, corresponding to that particular image
		'''
		self.data_list = self._makedata(self.root_dir)
		# # if self.cfg.TRAIN.DATASET_LENGTH != None:
		# # 	self.data_list = data_list[:self.cfg.TRAIN.DATASET_LENGTH]
		# else:
		# 	self.data_list = data_list
		

	

	def _read_label(self, file_name):
		ob_list = []

		# labels_dict = {'Car':0, 'Van':1, 'Truck':2, 'Tram':3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6}
		class_labels = self.cfg.INPUT.LABELS_TO_TRAIN
		
		box_list = []
		class_list = []
		with open(file_name) as file:
			objects = file.read().splitlines()
			for obj in objects:
				obj = obj.split()
				if obj[0] in class_labels:
					class_list.append(class_labels.index(obj[0]))
					box_list.append([float(i) for i in obj[4:8]])

		return box_list, class_list

	def _makedata(self, root_dir):
		data= []
		image_names = glob.glob(root_dir+"/images/training/*.png")

		# if self.cfg.TRAIN.DATASET_LENGTH != None:
		# 	image_names = image_names[:self.cfg.TRAIN.DATASET_LENGTH]
		# print(image_names)
		i = 0
		for name in image_names:
			img_size = Image.open(name).size
			if  img_size == (1242,375) and i<self.cfg.TRAIN.DATASET_LENGTH:
				# print("Got here")
				label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
				bbox_list, class_list = self._read_label(label_name)
				# print(len(bbox_list), len(class_list))
				if len(bbox_list)!=0:
					# print(name[-10:], " : Yes got an Object")
					data_point = Instances(img_size[::-1], gt_boxes=Boxes(torch.tensor(bbox_list)), gt_classes=torch.tensor(class_list))
					# data_point = Instances(img_size[::-1], gt_boxes=bbox_list, gt_classes=class_list)
					data.append({"image_path":name, "target": data_point })
					i+=1 
		return data	

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = self.data_list[idx]

		## loading the image
		img = Image.open(sample["image_path"]).convert('RGB')
		
		# transform the image
		if self.transform:
			img = self.transform(img)

		# sample["target"].toTensor() 
		# sample["target"].gt_boxes = Boxes(sample["target"].gt_boxes)
		sample["image"] = img

		return sample

#Gives Memory error this way
class KittiDatasetDict(Dataset):
	"""Kitti Dataset Reader."""

	def __init__(self, root_dir,transform=None, cfg = None):
		"""
		Args:
			root_dir (string): Path to the dataset.

			transform (callable, optional): Optional transform to be applied
				on a sample.

			cfg: config file
		"""

		self.cfg = cfg
		self.root_dir = root_dir
		self.transform = transform
		
		## has all the annotations 
		'''
		It's a dictionary, with keys() as absolute image_paths.
		Each key leads to a list of annotations, corresponding to that particular image
		'''
		self.data_dict = self._makedata(self.root_dir)
		# # if self.cfg.TRAIN.DATASET_LENGTH != None:
		# # 	self.data_list = data_list[:self.cfg.TRAIN.DATASET_LENGTH]
		# else:
		# 	self.data_list = data_list
		

	

	def _read_label(self, file_name):
		ob_list = []

		# labels_dict = {'Car':0, 'Van':1, 'Truck':2, 'Tram':3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6}
		class_labels = self.cfg.INPUT.LABELS_TO_TRAIN
		
		box_list = []
		class_list = []
		with open(file_name) as file:
			objects = file.read().splitlines()
			for obj in objects:
				obj = obj.split()
				if obj[0] in class_labels:
					class_list.append(class_labels.index(obj[0]))
					box_list.append([float(i) for i in obj[4:8]])

		return box_list, class_list

	def _makedata(self, root_dir):
		data= {}
		image_names = glob.glob(root_dir+"/images/training/*.png")

		# if self.cfg.TRAIN.DATASET_LENGTH != None:
		# 	image_names = image_names[:self.cfg.TRAIN.DATASET_LENGTH]
		# print(image_names)
		i = 0
		for name in image_names:
			img_size = Image.open(name).size
			if  img_size == (1242,375) and i<self.cfg.TRAIN.DATASET_LENGTH:
				# print("Got here")
				label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
				bbox_list, class_list = self._read_label(label_name)
				# print(len(bbox_list), len(class_list))
				if len(bbox_list)!=0:
					# print(name[-10:], " : Yes got an Object")
					data_point = Instances(img_size[::-1], gt_boxes=Boxes(torch.tensor(bbox_list)), gt_classes=torch.tensor(class_list))
					# data_point = Instances(img_size[::-1], gt_boxes=bbox_list, gt_classes=class_list)
					data[i] = {"image_path":name, "target": data_point }
					i+=1 
		return data	

	def __len__(self):
		return len(self.data_dict)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = self.data_dict[idx]

		## loading the image
		img = Image.open(sample["image_path"]).convert('RGB')
		
		# transform the image
		if self.transform:
			img = self.transform(img)

		# sample["target"].toTensor() 
		# sample["target"].gt_boxes = Boxes(sample["target"].gt_boxes)
		sample["image"] = img

		return sample

class KittiDataset(Dataset):
	"""Kitti Dataset Reader."""

	def __init__(self, root_dir,transform=None, cfg = None):
		"""
		Args:
			root_dir (string): Path to the dataset.

			transform (callable, optional): Optional transform to be applied
				on a sample.

			cfg: config file
		"""

		self.cfg = cfg
		self.root_dir = root_dir
		self.transform = transform
		
		## has all the annotations 
		'''
		It's a dictionary, with keys() as absolute image_paths.
		Each key leads to a list of annotations, corresponding to that particular image
		'''
		self.data_dict = self._makedata(self.root_dir)
		self.data_keys = list(self.data_dict.keys())
		# # if self.cfg.TRAIN.DATASET_LENGTH != None:
		# # 	self.data_list = data_list[:self.cfg.TRAIN.DATASET_LENGTH]
		# else:
		# 	self.data_list = data_list
		

	

	def _read_label(self, file_name):
		ob_list = []

		# labels_dict = {'Car':0, 'Van':1, 'Truck':2, 'Tram':3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6}
		class_labels = self.cfg.INPUT.LABELS_TO_TRAIN
		
		box_list = []
		class_list = []
		with open(file_name) as file:
			objects = file.read().splitlines()
			for obj in objects:
				obj = obj.split()
				if obj[0] in class_labels:
					class_list.append(class_labels.index(obj[0]))
					box_list.append([float(i) for i in obj[4:8]])

		return box_list, class_list

	def _makedata(self, root_dir):
		data= {}
		image_names = glob.glob(root_dir+"/images/training/*.png")

		# if self.cfg.TRAIN.DATASET_LENGTH != None:
		# 	image_names = image_names[:self.cfg.TRAIN.DATASET_LENGTH]
		# print(image_names)
		i = 0
		for name in image_names:
			img_size = Image.open(name).size
			if  img_size == (1242,375) and i<self.cfg.TRAIN.DATASET_LENGTH:
				# print("Got here")
				label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
				bbox_list, class_list = self._read_label(label_name)
				# print(len(bbox_list), len(class_list))
				if len(bbox_list)!=0:
					# print(name[-10:], " : Yes got an Object")
					data_point = Instances(img_size[::-1], gt_boxes=Boxes(torch.tensor(bbox_list)), gt_classes=torch.tensor(class_list))
					# data_point = Instances(img_size[::-1], gt_boxes=bbox_list, gt_classes=class_list)
					data[name] = data_point
					i+=1 
		return data	

	def __len__(self):
		return len(self.data_dict)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample= {}
		img_path = self.data_keys[idx]
		sample["image_path"] = img_path
		sample["target"] = self.data_dict[img_path]
		## loading the image
		img = Image.open(sample["image_path"]).convert('RGB')
		
		# transform the image
		if self.transform:
			img = self.transform(img)

		sample["image"] = img

		return sample



# A collate function to enable loading the kitti labels in batch
def kitti_collate_fn(batch): #batch is the list of samples

	elem = batch[0]
	batch = {key:[x[key] for x in batch] for key in elem}
	batch["image"] = torch.stack(batch["image"], dim=0)

	return batch

