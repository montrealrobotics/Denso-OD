import numpy as np
import torch
import torchvision
import torch.nn as nn

nms_thres = 0.7 

class NMS(object):
    """docstring for NMS"""
    def __init__(self, nms_thres = 0.7):
        super(NMS, self).__init__()
        self.thres = nms_thres
    
    def nms(self, reg, scores):
        reg = reg.numpy()
        scores = scores.numpy()

        x1 = reg[:, 0]
        y1 = reg[:, 1]
        x2 = reg[:, 2]
        y2 = reg[:, 3]
        scores = scores[:, 1]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order.item(0)
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.thres)[0]
            order = order[inds + 1]

        return torch.IntTensor(keep)

