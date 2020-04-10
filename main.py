import torch
import os
import sys
import numpy as np
import math
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.image as mpimg ## To load the image
import matplotlib.pyplot as plt
from torch import optim
from torchvision import transforms as T

from src.architecture import FasterRCNN
from src.config import Cfg as cfg # Configuration file
from src.datasets import Dataloader
from src.utils import utils, Boxes
from src.tools.trainer import Solver

from torchvision import datasets as dset
import torchvision
from torch.utils import tensorboard

import time

import argparse
# matplotlib.use('agg')

#----- Initial paths setup and loading config values ------ #

print("\n--- Setting up the training/testing \n")

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--name", required = True, help="Comments for the experiment")
ap.add_argument("-mode", "--mode",required = True, choices=['train', 'test'])
ap.add_argument("-weights", "--weights",default = None)
ap.add_argument("-resume", "--resume", default = False)
ap.add_argument("-epoch", "--epoch")
args = ap.parse_args()

exp_name = args.name
mode = args.mode

torch.manual_seed(cfg.RANDOMIZATION.SEED)
np.random.seed(cfg.RANDOMIZATION.SEED)

cfg.merge_From_file("config.yaml")
cfg.freeze()


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
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


model.train() if mode=="train" else model.eval()

#---------Training/Testing Cycle-----------#
epochs = cfg.TRAIN.EPOCHS
saving_freq = cfg.TRAIN.SAVE_MODEL_EPOCHS
solver = Solver(cfg, mode, args)
solver.train(epochs, saving_freq)


#-----------------------------------------------#


# #--------Seting up Training/Testing-------#

# if mode=='train':
#     print("--- Training Mode\n")
#     is_training = True
#     checkpoint = None

#     if not path.exists(experiment_dir):
        


#         if args.weights:
#             print("    :Using Pretrained wights: {} \n".format(args.weights))
#             checkpoint = torch.load(args.weights)

#     elif args.resume:
#         print("    :Resuming the training \n")
#         epoch = args['epoch'] #With which epoch you want to resume the training.
#         checkpoint = torch.load(model_save_dir + "/epoch_" +  str(epoch).zfill(5) + '.model')
#         cfg = checkpoint['cfg']

# else:
#     print("--- Testing Mode\n")
#     is_training = False
#     epoch = args.epoch #Wit which epoch you want to test the model
#     model_path = model_save_dir + "/epoch_" +  str(epoch).zfill(5) + '.model'
#     print("    : Using Model {}".format(model_path))
#     checkpoint = torch.load(model_path)
#     # cfg = checkpoint['cfg']

# #-----------------------------------------------#


# #---------Modelling and Trainer Building------#



# if cfg.TRAIN.OPTIM.lower() == 'adam':
#     optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
# elif cfg.TRAIN.OPTIM.lower() == 'sgd':
#     optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0.01)
# else:
#     raise ValueError('Optimizer must be one of \"sgd\" or \"adam\"')

# lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.MILESTONES,
#                 gamma=cfg.TRAIN.LR_DECAY, last_epoch=-1)

# if checkpoint:
#     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

# model.train() if mode=="train" else model.eval()
# #-----------------------------------------------#
