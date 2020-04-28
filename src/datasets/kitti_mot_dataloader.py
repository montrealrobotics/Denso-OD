import torch
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset #Default dataloader class for Pytorch
import os
import glob
from ..utils import Boxes, Instances
import time


class KittiMOTDataset(Dataset):
    """Kitti Dataset Reader."""

    def __init__(self, root_dir,tracks="0001", transform=None, cfg = None):
        """
        Args:
            root_dir (string): Path to the dataset.

            transform (callable, optional): Optional transform to be applied
                on a sample.

            cfg: config file
        """
        self.cfg = cfg
        self.transform = transform

        ## has all the annotations
        '''
        It's a dictionary, with keys() as absolute image_paths.
        Each key leads to a list of annotations, corresponding to that particular image
        '''
        self.data_dict = self._makedata(root_dir, tracks)
        self.data_keys = list(self.data_dict.keys())


    def _makedata(self, root_dir, tracks=["0001"]):
        data= {}
        for track in tracks:
            image_names = glob.glob(os.path.join(root_dir,"training/image_02",track,"/*.png"))
            image_names.sort(key=lambda f: int(f[-8:-4]))
            label_name = root_dir+"/training/label_02/"+track+".txt"
            object_rows = open(label_name).read().splitlines()
            class_labels = self.cfg.INPUT.LABELS_TO_TRAIN

            for i,name in enumerate(image_names):
                box_list = []
                class_list = []
                track_list = []
                img_size = Image.open(name).size
                if img_size == (1242,375) and i<self.cfg.TRAIN.DATASET_LENGTH:
                    for j,obj in enumerate(object_rows):
                        obj = obj.split()
                        if i==int(obj[0]):
                            if obj[2] in class_labels:
                                box_list.append([float(i) for i in obj[6:10]])
                                class_list.append(class_labels.index(obj[2]))
                                track_list.append(int(obj[1]))
                        else:
                            if len(box_list)!=0:
                                data_point = Instances(img_size[::-1], gt_boxes=Boxes(torch.tensor(box_list)), gt_classes=torch.tensor(class_list), gt_trackid = torch.tensor(track_list))
                                data[name] = data_point
                            object_rows = object_rows[j:]
                            break
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


class KittiMOTDataset_KF(Dataset):
    """Kitti Dataset Reader."""

    def __init__(self, root_dir,tracks="0001", transform=None, cfg = None):
        """
        Args:
            root_dir (string): Path to the dataset.

            transform (callable, optional): Optional transform to be applied
                on a sample.

            cfg: config file
        """
        self.cfg = cfg
        self.transform = transform
        self.seq_len = cfg.TRAIN.SEQUENCE_LENGTH
        step = cfg.TRAIN.STEP_BETWEEN_FRAME
        ## has all the annotations
        '''
        It's a dictionary, with keys() as absolute image_paths.
        Each key leads to a list of annotations, corresponding to that particular image
        '''
        self.data_list = self._makedata(root_dir, tracks)
        self.data_list = self.sample_data(self.data_list, self.seq_len, step)
        # self.data_keys = list(self.data_ dict.keys())
    
    def sample_data(self, tracks_data, seq_len, step):
        sequenced_data = []
        for track in tracks_data:
            sampled_points = np.array([np.arange(x,x+seq_len) for x in range(0,len(track)-2, step)])
            track = np.array(track)
            sequenced_data.append(track[sampled_points])

        # Contenate sequences from all tracks
        sequenced_data = np.concatenate(sequenced_data)

        #Remove any sequence with no boxes
        idx = [True if np.all([len(x) for x in seq]) else False for seq in sequenced_data]

        return sequenced_data[idx]

    def _makedata(self, root_dir, tracks=["0001"]):
        data_list = []
        for track in tracks:
            data= []
            image_names = glob.glob(os.path.join(root_dir,"image_02",track,"*.png"))
            image_names.sort(key=lambda f: int(f[-8:-4]))
            label_name = os.path.join(root_dir, "label_02", track+".txt")
            object_rows = open(label_name).read().splitlines()
            class_labels = self.cfg.INPUT.LABELS_TO_TRAIN
            for i,name in enumerate(image_names):
                box_list = []
                class_list = []
                track_list = []
                img_size = Image.open(name).size
                if img_size == (1242,375) and i<self.cfg.DATASET.LENGTH:
                    for j,obj in enumerate(object_rows):
                        obj = obj.split()
                        if i==int(obj[0]):
                            if obj[2] in class_labels:
                                box_list.append([float(k) for k in obj[6:10]])
                                class_list.append(class_labels.index(obj[2]))
                                track_list.append(int(obj[1]))
                        else:
                            data_point = Instances(img_size[::-1], image_path=name, gt_boxes=Boxes(torch.tensor(box_list)), 
                                        gt_classes=torch.tensor(class_list), gt_trackid = torch.tensor(track_list))
                            data.append(data_point)
                            object_rows = object_rows[j:]
                            break
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample= {}
        data_point = self.data_list[idx]
        sample["image_path"] = np.array([x.image_path for x in data_point])
        sample["target"] = data_point
        ## loading the image
        images = []

        for path in sample["image_path"]:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # transform the image

        sample["image"] = torch.stack(images)

        return sample

    # A collate function to enable loading the kitti labels in batch
    #batch is the list of samples
    def collate_fn(self, batch):
        elem = batch[0] # an sequence
        # seq_len = len(elem[elem.keys()[0]])
        batch = [{key:[x[key][i] for x in batch] for key in elem} for i in range(self.seq_len)]
        for x in batch:
            x["image"] = torch.stack(x["image"])

        return batch

    

