import torch

from .mAP.detection_map import DetectionMAP
from ..utils import Matcher, pairwise_iou, utils

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

        error = torch.mean((target.gt_boxes.tensor - instance.pred_boxes.tensor)**2, 0)
        sigma = torch.mean(instance.pred_variance, 0)

        self.sigma.append(sigma)
        self.error.append(error)

    def print(self):
        print("Caliberation Error: ", torch.mean(torch.abs(torch.stack(self.sigma)-torch.stack(self.error)), 0).cpu().numpy())
        

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

    def print(self):
        self.calib_error.print()
        self.mAP.plot()