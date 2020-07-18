from src.config import Cfg as cfg  # Configuration file
import numpy as np
import torch
import random 

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import sys

import matplotlib
import argparse
import time

import torchvision
from torch.utils import tensorboard

from src.engine.trainer import BackpropKF_Solver, General_Solver

matplotlib.use('agg')

#----- Initial paths setup and loading config values ------ #

print("\n--- Setting up Trainer/Tester \n")

ap = argparse.ArgumentParser()
ap.add_argument("-name",
                "--name",
                required=True,
                help="Comments for the experiment")
ap.add_argument("-config",
                "--config",
                required=False,
                default=None,
                help="Give your experiment configuration file")
ap.add_argument("-mode", "--mode", required=True, choices=['train', 'test'])
ap.add_argument("-weights", "--weights", default=None)
ap.add_argument("-resume", "--resume", default=False)
ap.add_argument("-epoch", "--epoch")
args = ap.parse_args()

if args.config:
    print("Loading exp config")
    cfg.merge_from_file(args.config)
cfg.freeze()


mode = args.mode

# Config Operations

#---------Training/Testing Cycle-----------#
epochs = cfg.TRAIN.EPOCHS
saving_freq = cfg.TRAIN.SAVE_MODEL_EPOCHS
# solver = BackpropKF_Solver(cfg, mode, args)
solver = General_Solver(cfg, mode, args)
if mode == "train":
    solver.train(epochs, saving_freq)
else:
    solver.test()

#-----------------------------------------------#
