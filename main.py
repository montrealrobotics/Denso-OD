import os
import sys

import numpy as np
import matplotlib
import argparse
import time

import torch
import torchvision
from torch.utils import tensorboard

from src.config import Cfg as cfg # Configuration file
from src.engine.trainer import BackpropKF_Solver

matplotlib.use('agg')

#----- Initial paths setup and loading config values ------ #

print("\n--- Setting up Trainer/Tester \n")

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--name", required = True, 
                help="Comments for the experiment")
ap.add_argument("-config", "--config", required=True,
                help="Give your experiment configuration file")
ap.add_argument("-mode", "--mode",required = True, 
                choices=['train', 'test'])
ap.add_argument("-weights", "--weights",default = None)
ap.add_argument("-resume", "--resume", default = False)
ap.add_argument("-epoch", "--epoch")
args = ap.parse_args()


cfg.merge_from_file(args.config)
cfg.freeze()

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

mode = args.mode

# Config Operations

#---------Training/Testing Cycle-----------#
epochs = cfg.TRAIN.EPOCHS
saving_freq = cfg.TRAIN.SAVE_MODEL_EPOCHS
solver = BackpropKF_Solver(cfg, mode, args)
solver.train(epochs, saving_freq)

#-----------------------------------------------#
