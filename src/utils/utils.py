import torch
import os
import sys
import numpy as np
import math
import argparse
from PIL import Image, ImageDraw
import matplotlib.image as mpimg ## To load the image
from torch import optim
import os.path as path
from torchvision import transforms as T
from src.config import Cfg as cfg

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .projection import ground_project
from .visualizer import Visualizer

def draw_instances(image, instances):
    class_labels = cfg.INPUT.LABELS_TO_TRAIN

    drawer = ImageDraw.Draw(image, mode=None)

    for instance in instances:
        box = instance.pred_boxes.tensor.numpy()[0]
        drawer.rectangle(box, outline ='red' ,width=3)
        drawer.text([box[0], box[1]-10],"{}: {:.2f}%".format(class_labels[instance.pred_classes.numpy()],
            instance.scores.numpy()), outline='green')

        if instance.has("pred_sigma"):
            sigma = np.sqrt(instance.pred_sigma.numpy())
            drawer.ellipse([box[0]-2*sigma[0], box[1]-2*sigma[1], box[0]+2*sigma[0], box[1]+2*sigma[1]], outline='blue', width=3)
            drawer.ellipse([box[2]-2*sigma[2], box[3]-2*sigma[3], box[2]+2*sigma[2], box[3]+2*sigma[3]], outline='blue', width=3)

    output = image.resize((np.array(image.size)/1.5).astype(int))

    return np.asarray(output)

def image_transform(cfg):

    """
    Input:
    cfg: configuration params
    img: Image loaded using matplotlib, numpy array of size H x W x C, in RGB format.
    (Note: if you use opencv to load image, convert it to RGB, as OpenCV works with BGR format)

    Output:
    torch.tensor : CxHxW  nomralised by mean and std of dataset
    """

    '''
    ### ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor  of shape (C x H x W) in the range [0.0, 1.0]

    '''

    transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)])

    return transform

def toPIL(img: torch.Tensor):
    ### T.ToPILImage() Converts an Tensor or numpy array in range [0, 1] with shape of (C x H x W) into an PIL image with range [0,255]
    return T.Compose([T.Normalize( mean=[-mean/std for mean, std in zip(cfg.INPUT.MEAN, cfg.INPUT.STD)],
            std=[1.0/x for x in cfg.INPUT.STD]), T.ToPILImage()])(img)

def toNumpyImage(img: torch.Tensor):
    return T.Normalize( mean=[-mean/std for mean, std in zip(cfg.INPUT.MEAN, cfg.INPUT.STD)],
            std=[1.0/x for x in cfg.INPUT.STD])(img).mul(255).numpy().astype('uint8')

def tb_logger(images, tb_writer, rpn_proposals=None, instances=None, name="Image"):

    image = toPIL(images[2].cpu())

    image_grid = image.resize((np.array(image.size)/1.5).astype(int))

    if rpn_proposals:
        proposal_locs = rpn_proposals[2].proposal_boxes[:50].tensor.cpu().numpy()
        proposal_img = draw_bbox(image.copy(), proposal_locs)

        image_grid = np.concatenate([image_grid, proposal_img], axis=1)

    if instances:
        pred = instances[2].pred_boxes.tensor.cpu().numpy()
        if instances[2].has("pred_sigma"):
            sigma = instances[2].pred_sigma.cpu().numpy()
        else:
            sigma=None
        img_cls = instances[2].pred_classes.cpu().numpy()

        prediction_img = draw_bbox(image.copy(), pred, img_cls, sigma)
        image_grid = np.concatenate([image_grid, prediction_img], axis=1)

    tb_writer.add_image(name, image_grid, dataformats='HWC')

def disk_logger(images, direc, rpn_proposals=None, instances=None, image_paths=None):
    images = images.cpu()

    for image, rpn_proposal, instance, path in zip(images, rpn_proposals, instances, image_paths):
        image = toPIL(image)
        img_visualizer = Visualizer(image, path, rpn_proposal.to("cpu"), instance.to("cpu"), cfg)

        if instance:
            img_visualizer.draw_instances()
            img_visualizer.draw_projection()

        img_visualizer.save(direc)
        print("{} written to disk".format(path[-10:]))
