import os
import torch

import numpy as np
from scipy.stats.stats import pearsonr   
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from .mAP.detection_map import DetectionMAP
from ..utils import Matcher, pairwise_iou, utils

class Evaluator(object):
    """Class for evaluating different metrics"""

    def __init__(self, num_classes):
        super(Evaluator, self).__init__()
        self.mAP = DetectionMAP(num_classes)
        self.calib_error = Caliberation_Error()
    
    def evaluate(self, images, instances, targets):
        for image, instance, target in zip(images, instances, targets):
            self.calib_error.evaluate(image, instance, target)
            instance = instance.numpy()
            target = target.numpy()
            self.mAP.evaluate(instance.pred_boxes,
                         instance.pred_classes,
                         instance.scores, 
                         target.gt_boxes,
                         target.gt_classes)
    
    def plot(self, direc):
        self.calib_error.plot(direc)

    def print(self):
        self.calib_error.print()
        self.mAP.plot()

        
class Caliberation_Error(object):
    """docstring for Caliberation_Error"""
    def __init__(self):
        super(Caliberation_Error, self).__init__()
        self.sigma = []
        self.error = []
        self.matcher = Matcher(
            [0.9],
            [0,1],
            allow_low_quality_matches=False,
        )

    def evaluate(self,image, instance, target):
        match_quality_matrix = pairwise_iou(
                                target.gt_boxes, instance.pred_boxes)

        matched_idxs, matched_labels = self.matcher(match_quality_matrix)
        target = target[matched_idxs]
        
        fg_inds = torch.nonzero(matched_labels==1).squeeze(1)
        
        target = target[fg_inds]
        instance = instance[fg_inds]

        # print(target.gt_boxes, instance.pred_boxes)
        # utils.single_disk_logger(image, target)

        error = torch.mean(torch.abs(target.gt_boxes.tensor - instance.pred_boxes.tensor), 1)
        sigma = torch.mean(torch.sqrt(instance.pred_variance), 1)

        # print(error)
        # print(error<20, sigma<20)

        # equivalent to error<20 and sigma<20
        indx = (error<20) & (sigma<20)

        self.sigma.append(sigma[indx])
        self.error.append(error[indx])

    def print(self):
        print("Caliberation Error: ", torch.mean(torch.abs(torch.cat(self.sigma)-torch.cat(self.error)), 0).cpu().numpy())
        corr, _ = pearsonr(torch.cat(self.sigma).cpu().numpy(),torch.cat(self.error).cpu().numpy())
        print("Correlation: ", corr)
    
    def plot(self, direc):
        errors = torch.cat(self.error).cpu().numpy()
        sigmas = torch.cat(self.sigma).cpu().numpy()

        bins = np.linspace(-5, 15, 20)
        # Draw the density plots
        sns.distplot(errors, hist = True, kde = True,
                     bins = bins,
                     kde_kws = {'linewidth': 3, 'clip':(bins.min(), bins.max())},
                     label = 'error')
        sns.distplot(sigmas, hist = True, kde = True,
                     bins = bins,
                     kde_kws = {'linewidth': 3, 'clip':(bins.min(), bins.max())},
                     label = 'var')

        plt.title("Error-Sigma Distribution")
        
        plt.savefig(os.path.join(direc, "sigma-error-dis"+".png"))
        plt.clf()

        sns.jointplot(errors, sigmas, kind="kde", 
            space=0, color="b").set_axis_labels("error", "sigma")
        plt.title("Error-Sigma Correlation")
        
        plt.savefig(os.path.join(direc, "sigma-error-corr"+".png"))
        plt.clf()
        

