import torch
import torchvision.ops as ops
import torch.nn as nn

class Detector(nn.Module):
    """docstring for Detector"""
    def __init__(self, input_size, hidden_size, class_size, pool_size, subsampling_ratio):
        super(Detector, self).__init__()
        roi_feat_size = (pool_size, pool_size)
        self.ROI = ops.RoIPool(roi_feat_size, subsampling_ratio)
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.class_out = nn.Sequential(nn.Linear(hidden_size, class_size), nn.Softmax())
        self.bbox = nn.Linear(hidden_size, class_size)
        
    def forward(feature_map, boxes):
        num_proposals = boxes[0]
        proposals = self.ROI(feature_map, boxes) # shape of proposals: Tensor[K, C, output_size[0], output_size[1]]
        proposals = proposals.view(num_proposals, -1)
        proposals = proposals.unsqueeze(0)  # output shape: [1, num_proposals, feat_size] here 1 assumes that batchsize is 1
        out = self.fc1(proposals)
        classes = self.class_out(out)
        bbox_out = self.bbox(out)

        return classes, bbox_out