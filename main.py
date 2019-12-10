import torch
import os
import sys
import numpy as np
import math
import argparse
from PIL import Image, ImageDraw
import matplotlib.image as mpimg ## To load the image
import matplotlib.pyplot as plt
from torch import optim
import os.path as path
from torchvision import transforms as T
## Inserting path of src directory
# sys.path.insert(1, '../')

from src.architecture import FasterRCNN
from src.config import Cfg as cfg # Configuration file
from src.datasets import process_kitti_labels
from src.datasets import kitti_collate_fn
from src.datasets import KittiDataset # Dataloader
from src.utils import utils, Boxes
from src.tools import train_test
# from src.pytorch_nms import nms as NMS


from torchvision import datasets as dset
from torchvision import transforms as T
import torchvision
from torch.utils import tensorboard

import time


#----- Initial paths setup and loading config values ------ #

print("\n--- Setting up the training/testing \n")

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")
ap.add_argument("-mode", "--mode",required = True, choices=['train', 'test'])
ap.add_argument("-weights", "--weights",default = None)
ap.add_argument("-resume", "--resume", default = False)
ap.add_argument("-epoch", "--epoch")
args = ap.parse_args()

experiment_dir = cfg.PATH.LOGS + "/" + args.experiment_comment
mode = args.mode
results_dir = experiment_dir+"/results"
graph_dir = experiment_dir+"/tf_summary"
model_save_dir = experiment_dir+"/models"

torch.manual_seed(cfg.RANDOMIZATION.SEED)
np.random.seed(cfg.RANDOMIZATION.SEED)

device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
print("Using the device for training: {} \n".format(device))


#-----------------------------------------------#

#--------Seting up Training/Testing-------#

if mode=='train':
    print("--- Training Mode\n")
    is_training = True
    checkpoint = None

    if not path.exists(experiment_dir):
        os.mkdir(experiment_dir)
        os.mkdir(results_dir)
        os.mkdir(model_save_dir)
        os.mkdir(graph_dir)


        if args.weights:
            print("    :Using Pretrained wights: {} \n".format(args.weights))
            checkpoint = torch.load(args.weights)
    
    elif args.resume:
        print("    :Resuming the training \n")
        epoch = args['epoch'] #With which epoch you want to resume the training. 
        checkpoint = torch.load(model_save_dir + "/epoch_" +  str(epoch).zfill(5) + '.model')
        cfg = checkpoint['cfg']

else:
    print("--- Testing Mode\n")
    is_training = False
    epoch = args.epoch #Wit which epoch you want to test the model
    model_path = model_save_dir + "/epoch_" +  str(epoch).zfill(5) + '.model'
    print("    : Using Model {}".format(model_path))
    checkpoint = torch.load(model_path)
    cfg = checkpoint['cfg']

#-----------------------------------------------#


#---------Modelling and Trainer Building------#

print("--- Building the Model \n")
# print(cfg.ROI_HEADS.SCORE_THRESH_TEST)
model = FasterRCNN(cfg)
model = model.to(device)

if cfg.TRAIN.OPTIM.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
elif cfg.TRAIN.OPTIM.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0.01)
else:
    raise ValueError('Optimizer must be one of \"sgd\" or \"adam\"')

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.MILESTONES, 
                gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)

if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


model.train() if mode=="train" else model.eval()

tb_writer = tensorboard.SummaryWriter(graph_dir)
#-----------------------------------------------#


#-------- Dataset loading and manipulation-------#
dataset_path = cfg.PATH.DATASET

if not path.exists(dataset_path):
    print("Dataset path doesn't exist")

transform = utils.image_transform(cfg) # this is tranform to normalise/standardise the images

if mode=="train":
    print("--- Loading Training Dataset \n ")

    dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg) #---- Dataloader
    dataset_len = len(dataset)
    ## Split into train & validation
    train_len = int(cfg.TRAIN.DATASET_DIVIDE*dataset_len)
    val_len = dataset_len - train_len

    print("--- Data Loaded---")
    print("Number of Images in Dataset: {} \n".format(dataset_len))
    print("Number of Classes in Dataset: {} \n".format(cfg.INPUT.NUM_CLASSES))

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)


else:
    print("---Loading Test Dataset \n")
    dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg) #---- Dataloader

    dataset_len = len(dataset)
    ## Split into train & validation
    train_len = int(cfg.TRAIN.DATASET_DIVIDE*dataset_len)
    val_len = dataset_len - train_len

    print("--- Data Loaded---")
    print("Number of Images in Dataset: {} \n".format(dataset_len))
    print("Number of Classes in Dataset: {} \n".format(cfg.INPUT.NUM_CLASSES))

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)
#----------------------------------------------#


#---------Training/Testing Cycle-----------#
epochs = cfg.TRAIN.EPOCHS

if mode=="train":
    print("Starting the training in 3.   2.   1.   Go \n")
    train_test.train(model, train_loader, val_loader, optimizer, epochs, tb_writer, lr_scheduler, device, model_save_dir, cfg)

if mode=='test':
    print("Starting the inference in 3.   2.   1.   Go \n")
    train_test.test(model, test_loader, device, results_dir)