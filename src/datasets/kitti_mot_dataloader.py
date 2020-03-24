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

    def __init__(self, root_dir,track="0001", transform=None, cfg = None):
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
        self.data_dict = self._makedata(root_dir, track)
        self.data_keys = list(self.data_dict.keys())
        # # if self.cfg.TRAIN.DATASET_LENGTH != None:
        # #     self.data_list = data_list[:self.cfg.TRAIN.DATASET_LENGTH]
        # else:
        #   self.data_list = data_list

    def _makedata(self, root_dir, track="0001"):
        data= {}
        image_names = glob.glob(root_dir + "/training/image_02/" + track + "/*.png")
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


# A collate function to enable loading the kitti labels in batch
def kitti_collate_fn(batch): #batch is the list of samples

    elem = batch[0]
    batch = {key:[x[key] for x in batch] for key in elem}
    batch["image"] = torch.stack(batch["image"], dim=0)

    return batch
