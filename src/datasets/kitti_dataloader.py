import torch
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset #Default dataloader class for Pytorch
import os
import glob


class KittiDataset(Dataset):
	"""NuScenes dataset for 2d annotations."""

	def __init__(self, root_dir, transform=None, cfg = None):
		"""
		Args:
			root_dir (string): Path to the dataset.

			transform (callable, optional): Optional transform to be applied
				on a sample.

			cfg: config file
		"""

		self.root_dir = root_dir

		### loading all the annotations!!
		annotations_dict = self._makedata(self.root_dir)
		self.transform = transform
		
		## has all the annotations 
		'''
		It's a dictionary, with keys() as absolute image_paths.
		Each key leads to a list of annotations, corresponding to that particular image
		'''
		self.annotation_dict = annotations_dict
		self.cfg = cfg
	

	def _read_label(self, file_name):
		ob_list = []
		class_list = ['Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Person_sitting', 'Cyclist']
		with open(file_name) as file:
			objects = file.read().splitlines()
			for obj in objects:
				em_dict = {}
				obj = obj.split()
				if obj[0] in class_list:
					# print(obj[0])
					em_dict['class'] = obj[0]
					em_dict['bbox'] = [float(i) for i in obj[4:8]]
					ob_list.append(em_dict)

		return ob_list

	def _makedata(self, root_dir):
		data_dict = {}
		image_names = glob.glob(root_dir+"/images/training/*.png")

		for name in image_names:
			label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
			objects = self._read_label(label_name)
			if len(objects)!=0:
				# print(name[-10:], " : Yes got an Object")
				data_dict[name] = objects
		
		return data_dict	

	def __len__(self):
		return len(self.annotation_dict.keys())

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img = None
		target = None
		img_path  = None

		img_path = list(self.annotation_dict.keys())[idx]

		## loading the image
		img = Image.open(img_path).convert('RGB')

		## Annotations
		target = self.annotation_dict[img_path]
		
		## transform the image
		if self.transform:
			img = self.transform(img)


		return img, target, img_path



# A collate function to enable loading the kitti labels in batch
def kitti_collate_fn(batch):

	## let's get the images stacked up(works with batchsize 1 as kitti has some images of diff sizes.)
	data = torch.stack([item[0] for item in batch])

	## getting the target in a list
	target = [item[1] for item in batch]

	## Getting paths in list
	paths = [item[2] for item in batch]
	# target = [item[1] for item in batch]
	# target = torch.LongTensor(target)
	return [data, target, paths]


