import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Function
from ..utils import Boxes, Instances
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms as nms

def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every image
    batch_idx = torch.arange(num_images, device=device)

    Hi_Wi_A = pred_objectness_logits.shape[1]
    num_proposals = min(pre_nms_topk, Hi_Wi_A)

    # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
    # topk_scores_i, topk_idx = pred_objectness_logits.topk(num_proposals, dim=1)
    sorted_objectness_logits, idx = pred_objectness_logits.sort(descending=True, dim=1)
    topk_scores = sorted_objectness_logits[:, :num_proposals] #N x topk
    
    topk_idx = idx[:, :num_proposals] # N x topk index
    topk_proposals = proposals[batch_idx[:, None], topk_idx]  # N x topk x 4
    level_ids = torch.full((num_proposals,), 0, dtype=torch.int64, device=device)


    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    # print(image_sizes)
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        # Commenting the below because of an error
        boxes.clip(image_size) 

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_side_len)
        lvl = level_ids
        # print(len(boxes))
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # keep = nms(boxes.tensor, scores_per_img, nms_thresh)

        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    
    return results

#Detectron2 implementation
def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def bayes_od_clustering(
        predicted_boxes_class_counts,
        predicted_boxes_means,
        predicted_boxes_covs,
        cluster_centers,
        affinity_matrix,
        affinity_threshold=0.7):
    """
    Bayesian NMS clustering to output a single probability distribution per object in the scene.

    Args:
        predicted_boxes_means: Nx4x1 tensor containing resulting means from mc_dropout. N is the number of anchors processed by the model.
        predicted_boxes_covs: Nx4x4 tensor containing resulting covariance matrices from mc_dropout.
        cat_count_res: NxC tensor containing the alpha counts from mc dropout. C is the number of categories.
        cluster_centers: Kx1 tensor containing the indices of centers of clusters based on minimum entropy. K is the number of clusters.
        affinity_matrix: NxN matrix containing affinity measures between bounding boxes.
        affinity_threshold: scalar to determine the minimum affinity for clustering.
        
    Returns:
        final_scores: Kx4 tensor containing the parameters of the final categorical posterior distribution describing the objects' categories.
        final_means: Kx4x1 tensor containing the mean vectors of the posterior gaussian distribution describing objects' location in the scene.
        final_covs: Kx4x4 tensor containing the covariance matrices of the posterior gaussian distribution describing objects' location in the scene.
    """

    # Initialize lists to save per cluster results
    final_box_means = []
    final_box_covs = []
    final_box_class_scores = []
    final_box_class_counts = []
    for cluster_center in cluster_centers:

        # Get bounding boxes with affinity > threshold with the center of the
        # cluster
        cluster_inds = affinity_matrix[:, cluster_center] > affinity_threshold
        cluster_means = predicted_boxes_means[cluster_inds, :, :]
        cluster_covs = predicted_boxes_covs[cluster_inds, :, :]

        # Compute mean and covariance of the final posterior distribution
        cluster_precs = np.array(
            [np.linalg.inv(member_cov) for member_cov in cluster_covs])

        final_cov = np.linalg.inv(np.sum(cluster_precs, axis=0))
        final_box_covs.append(final_cov)

        mean_temp = np.array([np.matmul(member_prec, member_mean) for
                              member_prec, member_mean in
                              zip(cluster_precs, cluster_means)])
        mean_temp = np.sum(mean_temp, axis=0)
        final_box_means.append(np.matmul(final_cov, mean_temp))

        # Compute the updated parameters of the categorical distribution
        final_counts = predicted_boxes_class_counts[cluster_inds, :]
        final_score = (final_counts) / \
            np.expand_dims(np.sum(final_counts, axis=1), axis=1)

        if final_score.shape[0] > 3:
            cluster_center_score = np.expand_dims(predicted_boxes_class_counts[cluster_center, :] / np.sum(
                predicted_boxes_class_counts[cluster_center, :]), axis=0)
            cluster_center_score = np.repeat(
                cluster_center_score, final_score.shape[0], axis=0)

            cat_ent = entropy(cluster_center_score.T, final_score.T)

            inds = np.argpartition(cat_ent, 3)[:3]

            final_score = final_score[inds]
            final_counts = final_counts[inds]

        final_score = np.mean(final_score, axis=0)
        final_counts = np.sum(final_counts, axis=0)
        final_box_class_scores.append(final_score)
        final_box_class_counts.append(final_counts)

    final_box_means = np.array(final_box_means)
    final_box_class_scores = np.array(final_box_class_scores)

    # 70 is a calibration parameter specific to BDD and KITTI datasets. It
    # gives the best PDQ. It does not affect AP or MUE.
    final_box_covs = np.array(final_box_covs) * 70
    final_box_class_counts = np.array(final_box_class_counts)

    return final_box_class_scores, final_box_means, final_box_covs, final_box_class_counts


