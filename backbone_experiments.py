'''
In this, we will load two different models and play around with it. 
'''


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
from src.eval.detection_map import DetectionMAP
import matplotlib.pyplot as plt

from torchvision import datasets as dset
import torchvision
from torch.utils import tensorboard

import time


device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
print("Using the device for training: {} \n".format(device))
dataset_path = cfg.PATH.DATASET
transform = utils.image_transform(cfg) # this is tranform to normalise/standardise the images
#-----------------------------------------------#

#--------Loading model 1 with resnet-34-------#
is_training = False
model_path = '/network/tmp1/bhattdha/Denso-kitti-probabilistic-models/logs/resnet-34_train/models/epoch_00050.model'
print("    : Using Model {}".format(model_path))
checkpoint_rn34 = torch.load(model_path)
cfg_resnet34 = checkpoint_rn34['cfg']
cfg_resnet34.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
model_rn34 = FasterRCNN(cfg_resnet34)
model_rn34 = model_rn34.to(device)
model_rn34.load_state_dict(checkpoint_rn34['model_state_dict'], strict=False)
model_rn34.eval()


#-----------------------------------------------#

#--------Loading model 2 with resnet-101-------#
is_training = False
model_path = '/network/tmp1/bhattdha/Denso-kitti-probabilistic-models/logs/resnet-101_train/models/epoch_00040.model'
print("    : Using Model {}".format(model_path))
checkpoint_rn101 = torch.load(model_path)
cfg_resnet101 = checkpoint_rn101['cfg']
cfg_resnet101.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
model_rn101 = FasterRCNN(cfg_resnet101)
model_rn101 = model_rn101.to(device)
model_rn101.load_state_dict(checkpoint_rn101['model_state_dict'], strict=False)
model_rn101.eval()

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

## Small run for fun

'''
Let's load the dataset
'''
dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg_resnet34) #---- Dataloader
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

## following are the times we need
m1_bb_st = 0
m1_bb_end = 0
m1_bb_av = []

m2_bb_st = 0
m2_bb_end = 0
m2_bb_av = []

m1_rpn_st = 0
m1_rpn_end = 0
m1_rpn_av = []

m1_stage1_tot = 0
m1_stage1_av = []

m2_stage2_st = 0
m2_stage2_end = 0
m2_stage2_av = []

tot_time = []

# stage_1_tot_time = 0
stage_2_st_time = 0
stage_2_tot_time = 0

## Let's test our way ;)
is_training= False
mAP = DetectionMAP(7)
with torch.no_grad():
    for idx, batch_sample in enumerate(test_loader):

        in_images = batch_sample['image'].to(device)
        print("shape is: ", in_images.shape[-2:])
        targets = [x.to(device) for x in batch_sample['target']]
        img_paths = batch_sample['image_path']


        '''
        Experiment 1: Stage 1: small backbone, stage 2: big backbone
        '''
        # rpn_proposals, instances, proposal_losses, detector_losses = model_rn34(in_images, targets, is_training)

        print(in_images.shape)

        ## Stage 1
        ###### Backbone of model-1
        m1_bb_st = time.time()
        feature_map_rn34 = model_rn34.backbone(in_images) # feature_map : [N, self.backbone_net.out_channels, H, W]
        m1_bb_end = time.time() - m1_bb_st
        m1_bb_av.append(m1_bb_end)

        ###### RPN of model-1
        m1_rpn_st = time.time()
        rpn_proposals_rn34, rpn_losses_rn34 = model_rn34.rpn(feature_map_rn34, targets, in_images.shape[-2:], is_training) # topK proposals sorted in decreasing order of objectness score and losses: []
        m1_rpn_end = time.time() - m1_rpn_st
        m1_rpn_av.append(m1_rpn_end)

        m1_stage1_tot = m1_bb_end + m1_rpn_end
        m1_stage1_av.append(m1_stage1_tot)

        ###### Backbone of model-2
        m2_bb_st = time.time()
        feature_map_rn101 = model_rn101.backbone(in_images) # feature_map : [N, self.backbone_net.out_channels, H, W]
        m2_bb_end = time.time() - m2_bb_st
        m2_bb_av.append(m2_bb_end)

        ## Stage 2
        m2_stage2_st = time.time()
        detections, detection_loss = model_rn101.detector(feature_map_rn101, rpn_proposals_rn34, targets, is_training)
        m2_stage2_end = time.time() - m2_stage2_st
        m2_stage2_av.append(m2_stage2_end)
 
        tot_time.append(m1_stage1_tot + m2_stage2_end)

        print("Yeets happening?")
        # rpn_proposals, instances, proposal_losses, detector_losses = model(in_images, targets, is_training)
        # print(time.time() - start)
        # utils.disk_logger(in_images, results_dir, rpn_proposals_rn34, detections, img_paths)
        # utils.ground_projection(in_images, results_dir, instances, img_paths)

        for instance, target in zip(detections, targets):
            pred_bb1 = instance.pred_boxes.tensor.cpu().numpy()
            pred_cls1 = instance.pred_classes.cpu().numpy() 
            pred_conf1 = instance.scores.cpu().numpy()
            gt_bb1 = target.gt_boxes.tensor.cpu().numpy()
            gt_cls1 = target.gt_classes.cpu().numpy()
            mAP.evaluate(pred_bb1, pred_cls1, pred_conf1, gt_bb1, gt_cls1)

mAP.plot()
plt.savefig('small_big.png')
plt.show()

print("backbone(stage-2) time is: ", np.mean(m2_bb_av))
print("backbone(stage-1) time is: ", np.mean(m1_bb_av))
print("RPN(stage-1) time is: ", np.mean(m1_rpn_av))
print("Stage-1 total time is: ", np.mean(m1_stage1_av))
print("Stage-2 total time is: ", np.mean(m2_stage2_av))
print("Total time is: ", np.mean(tot_time))

import ipdb; ipdb.set_trace()


# # model = FasterRCNN(cfg)
# # model = model.to(device)


# # import ipdb; ipdb.set_trace()
# #----- Initial paths setup and loading config values ------ #

# print("\n--- Setting up the training/testing \n")

# ap = argparse.ArgumentParser()
# ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")
# ap.add_argument("-mode", "--mode",required = True, choices=['train', 'test'])
# ap.add_argument("-weights", "--weights",default = None)
# ap.add_argument("-resume", "--resume", default = False)
# ap.add_argument("-epoch", "--epoch")
# args = ap.parse_args()

# experiment_dir = cfg.PATH.LOGS + "/" + args.experiment_comment
# mode = args.mode
# results_dir = experiment_dir+"/results"
# graph_dir = experiment_dir+"/tf_summary"
# model_save_dir = experiment_dir+"/models"

# torch.manual_seed(cfg.RANDOMIZATION.SEED)
# np.random.seed(cfg.RANDOMIZATION.SEED)

# device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
# print("Using the device for training: {} \n".format(device))


# #-----------------------------------------------#

# #--------Seting up Training/Testing-------#

# if mode=='train':
#     print("--- Training Mode\n")
#     is_training = True
#     checkpoint = None

#     if not path.exists(experiment_dir):
#         os.mkdir(experiment_dir)
#         os.mkdir(results_dir)
#         os.mkdir(model_save_dir)
#         os.mkdir(graph_dir)


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
#     cfg = checkpoint['cfg']

# #-----------------------------------------------#


# #---------Modelling and Trainer Building------#

# print("--- Building the Model \n")
# # print(cfg.ROI_HEADS.SCORE_THRESH_TEST)
# model = FasterRCNN(cfg)
# model = model.to(device)

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

# tb_writer = tensorboard.SummaryWriter(graph_dir)
# #-----------------------------------------------#


# #-------- Dataset loading and manipulation-------#
# dataset_path = cfg.PATH.DATASET

# if not path.exists(dataset_path):
#     print("Dataset path doesn't exist")

# transform = utils.image_transform(cfg) # this is tranform to normalise/standardise the images

# if mode=="train":
#     print("--- Loading Training Dataset \n ")

#     dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg) #---- Dataloader
#     dataset_len = len(dataset)
#     ## Split into train & validation
#     train_len = int(cfg.TRAIN.DATASET_DIVIDE*dataset_len)
#     val_len = dataset_len - train_len

#     print("--- Data Loaded---")
#     print("Number of Images in Dataset: {} \n".format(dataset_len))
#     print("Number of Classes in Dataset: {} \n".format(cfg.INPUT.NUM_CLASSES))

#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
#                 shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)
    
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
#                 shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)


# else:
#     print("---Loading Test Dataset \n")
#     dataset = KittiDataset(dataset_path, transform = transform, cfg = cfg) #---- Dataloader

#     dataset_len = len(dataset)
#     ## Split into train & validation
#     train_len = int(cfg.TRAIN.DATASET_DIVIDE*dataset_len)
#     val_len = dataset_len - train_len

#     print("--- Data Loaded---")
#     print("Number of Images in Dataset: {} \n".format(dataset_len))
#     print("Number of Classes in Dataset: {} \n".format(cfg.INPUT.NUM_CLASSES))

#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

#     test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
#                 shuffle=cfg.TRAIN.DSET_SHUFFLE, collate_fn = kitti_collate_fn, drop_last=True)
# #----------------------------------------------#


# #---------Training/Testing Cycle-----------#
# epochs = cfg.TRAIN.EPOCHS

# if mode=="train":
#     print("Starting the training in 3.   2.   1.   Go \n")
#     train_test.train(model, train_loader, val_loader, optimizer, epochs, tb_writer, lr_scheduler, device, model_save_dir, cfg)

# if mode=='test':
#     print("Starting the inference in 3.   2.   1.   Go \n")
#     train_test.test(model, test_loader, device, results_dir)