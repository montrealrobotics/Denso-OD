
'''
Class to generate targets for 
Region proposal network. 
'''

import torch
import sys
import numpy as np
from . import AnchorGenerator
from ..utils import Boxes, pairwise_iou, subsample_labels
from ..loss import rpn_losses

class RPNProcessing(object):

    def __init__(
        self,
        cfg,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        gt_boxes,
        matcher,
        box2box_transform,
        image_sizes=None
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.cfg = cfg
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.anchors = anchors #[N, num_of_anchor, 4] - These will be boxes class
        self.gt_boxes = gt_boxes # [batch_size, M, 4] M= number of gt boxes in an image - These will be Boxes class
        self.image_size = image_sizes
        self.num_images = len(gt_boxes)
        
        self.box2box_transform = box2box_transform
        self.anchor_matcher = matcher
        
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction  = cfg.RPN.POSITIVE_FRACTION
        self.boundary_threshold = cfg.RPN.BOUNDARY_THRESH
        self.smooth_l1_beta = cfg.RPN.SMOOTH_L1_BETA

    def get_rpn_target(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
    
        for gt_boxes_i, anchors_i in zip(self.gt_boxes, self.anchors):
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i) #[num_gt_boxes, num_anchors]

            # Acnhors are assosiated with their ground truths, matched_idxs: [num_of_anchor], 
            # each element is index of gt with which the anchor is matched
            # gt_objectness_logits_i: [num_of_achors], each anchor is labeled as 0,-1,1. 
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix) 

            if (self.boundary_threshold >= 0) & (self.image_size is not None):
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(self.image_size, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)

            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        # This generates the labels and bbox regression target.
        # gt_objectness_logits: [batch_size, num_anchors]  gt_anchor_deltas: [batch_size, num_anchors, 4]
        return gt_objectness_logits, gt_anchor_deltas 
            

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        # print(self.pred_objectness_logits.shape, self.pred_anchor_deltas.shape)
        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        # gt_objectness_logits: [batch_size, num_anchors]  gt_anchor_deltas: list of len batch_size and each element:[num_anchors, 4]   
        gt_objectness_logits, gt_anchor_deltas = self.get_rpn_target() 

        # Stack to: (N, num_anchors_per_image) 
        # In this gt_objectness_logits only 256 anchors in total have 1 and 0, others have been marked -1.
        # So those 256 are only gonna be used in training. 
        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits], dim=0
        )
       
        gt_objectness_logits = gt_objectness_logits.flatten() # shape: [batch_size*num_anchors_per_featmap=H*W*9] -> 1D tensor

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0) # shape: [batch_size, num_anchors, 4]
        gt_anchor_deltas = gt_anchor_deltas.reshape(-1, 4) # shape: [tot_anchors_batch, 4]

        # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
        pred_objectness_logits = self.pred_objectness_logits.permute(0, 2, 3, 1).flatten()

        # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
        #          -> (N*Hi*Wi*A, B)
        x = self.pred_anchor_deltas
        pred_anchor_deltas = x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).reshape(-1, 4)
                
        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )

        normalizer = 1.0 / (self.batch_size_per_image * self.cfg.TRAIN.BATCH_SIZE)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        # print(self.pred_anchor_deltas.shape)

        B = self.anchors[0].tensor.size(-1)
        N, _ , Hi, Wi = self.pred_anchor_deltas.shape

         # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)    
        pred_anchor_deltas = self.pred_anchor_deltas.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
        
        anchors = Boxes.cat(self.anchors)
        proposals = self.box2box_transform.apply_deltas(pred_anchor_deltas, anchors.tensor)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
        proposals = proposals.view(N, -1, B)
        
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        # print(self.pred_objectness_logits.shape)
        
        pred_objectness_logits = self.pred_objectness_logits.permute(0, 2, 3, 1).reshape(self.num_images, -1)
         
        return pred_objectness_logits


class RPN_targets(torch.nn.Module):
    """Generates target for region proposal network.(Useful for training)"""
    def __init__(self, cfg):
        super(RPN_targets, self).__init__()
        self.anchor_generator_obj = AnchorGenerator()
        self.cfg = cfg

    def get_targets(self, image, feature_map, targets):


        """
        Inputs: 
        Image: Image or batch of images as torch tensor
        feature_map: feature map output of backbone network
        Targets: Ground truth labels and bonding boxes corresponding to the image

        """
    
        anchors = self.anchor_generator_obj.get_anchors(image, feature_map, self.cfg) ## Nx4 numpy array
        # print(anchors.shape)
        orig_anchors = anchors

        ## Some of these anchors may not be valid
        ## Let's get indices of anchors which are inside the image
        im_height = image.size()[2]
        im_width = image.size()[3]

        inside_indices = np.where((anchors[:,0] >= 0) &
                             (anchors[:,1] >= 0) &
                             (anchors[:,2] <= im_height) &
                             (anchors[:,3] <= im_width))[0]

        # print("Invalid anchors are:" , len(orig_anchors) - len(inside_indices))
        # print(len(inside_indices))
        ## Constructing an array holding valid anchor boxes
        ## These anchors basically fall inside the image
        inside_anchors = anchors[inside_indices]

        '''
        Each anchor will either be positive or negative. Hence, we are creating 
        an array of length same as number of valid anchors, and we will assign them
        either 0(negative anchor) or 1(positive anchor) or -1(not valid)
        '''

        anchor_labels = np.empty((len(inside_indices), ), dtype=np.int32)
        anchor_labels.fill(-1)

        ious = self.compute_anchor_iou(inside_anchors, targets['boxes'])

        '''
        Now comes the part where we assign labels to anchors. 

        Positive anchors: 
            1. The anchors with highest IoU with the ground truth objects
            2. Anchor with IoU > 0.7 with the ground truth object.

        Negative anchors:
            1. All the anchors whose IoU with all the ground truth objects is lesser than 0.3,
               are negative anchors. 
        '''


        ## Finding highest IoU for each gt_box, and the corresponding anchor box
        ## Contains M indices in range [0, N], for M objects
        gt_argmax_ious = ious.argmax(axis=0) 
        # print(len(gt_argmax_ious))

        ## saving maximum ious, for all the ground truth objects, gives the maximun IoU
        ## Contains M IoUs, for M ground truth objects
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] 
        
        """
            This solves a very important bug. 
            A lot of times, it's possible that due to object's faulty
            annotation around the corners, we may not get proper maximum
            IoU, we may end up with 0 IoU, which can mess up the calculation
            ahead, so we identify such a case and don't let our network train 
            on that. Instead, we leave the function(effectively raising an exception
            in a training script, which results in moving to the next image)
        """
        if np.sum(gt_max_ious < 1e-9) > 0:
            return

        ## Finding maximum IoU for each anchor and its corresponding GT box
        ## Contains object label with highest IoU
        ## Contains N indices in range [0, M]
        argmax_ious = ious.argmax(axis=1) 
        # print(argmax_ious)

        ## saving maximum IoU for each anchor 
        ## Contains N IoUs, for N anchors. Highest IoU for each anchor
        max_ious = ious[np.arange(ious.shape[0]),argmax_ious] 

        ## we gotta find all the anchor indices with gt_max_ious
        ## Multiple anchors could have highest IoUs which we discovered earlier. 
        ## Contains ID of anchors with highest IoUs
        # print(ious.shape, gt_max_ious)

        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        # print(gt_argmax_ious)
        

        '''
        Let's assign labels! Important
        '''

        pos_anchor_iou_threshold = self.cfg.ANCHORS.POS_PROPOSAL_THRES
        neg_anchor_iou_threshold = self.cfg.ANCHORS.NEG_PROPOSAL_THRES



        ## IF max_iou for and anchor is lesser than neg_anchor_iou_threshold, it's a negative anchor.
        anchor_labels[max_ious < neg_anchor_iou_threshold] = 0      


        # print(np.max(max_ious))
        ## All the anchors with highest IoUs get to be positive anchors
        anchor_labels[gt_argmax_ious] = 1
        # print(len(gt_argmax_ious))
        
        # print(len(anchor_labels))
        ## All the anchors with iou greater than pos_anchor_iou_threshold deserve to be a positive anchor
        anchor_labels[max_ious >= pos_anchor_iou_threshold] = 1

        # print(len(anchor_labels[max_ious < neg_anchor_iou_threshold]))

        ## We don't use all anchors to compute loss. We sample negative and positive anchors 
        ## in 1:1 ratio to avoid domination of negative anchors.

        ## Ratio of positive and negative anchors
        pos_to_neg_ratio = 0.5
        num_of_anchor_samples = self.cfg.ANCHORS.TRAINING ## Total number of anchors

        n_pos = int(num_of_anchor_samples*pos_to_neg_ratio) ## Number of positive anchors
        n_neg = num_of_anchor_samples - n_pos          ## Number of negative anchors

        '''
        Sampling positive and negative anchors.

        Sometimes, it possible that number of positive anchors
        are lesser than number of samples required. In that case, 
        we keep all of them for our purpose. But if we have lesser 
        number of positive proposals than required, we also make
        number of negative proposals = number of positive proposals

        '''

        pos_anchor_indices = np.where(anchor_labels == 1)[0] ## Indices with positive label
        neg_anchor_indices = np.where(anchor_labels == 0)[0] ## Indices with negaitve label
        # print(len(anchor_labels), np.sum(anchor_labels == 0), np.sum(anchor_labels == 1), np.sum(anchor_labels == -1))
        # print("Number of negative anchors are: ", len(neg_anchor_indices))

        if len(pos_anchor_indices) > n_pos:
            disable_index = np.random.choice(pos_anchor_indices, size=(len(pos_anchor_indices) - n_pos), replace=False)
            anchor_labels[disable_index] = -1

        ### Important!!! ###
        '''
        If we have lesser number of positive anchors, we keep all of them. 
        But then we also sample lesser number of negative anchors. Because
        we need to keep their ratio proper. 
        '''

        if len(pos_anchor_indices) < n_pos:
            # n_neg = num_of_anchor_samples - len(pos_anchor_indices)
            n_neg = len(pos_anchor_indices)
            

        if len(neg_anchor_indices) > n_neg:
            disable_index = np.random.choice(neg_anchor_indices, size=(len(neg_anchor_indices) - n_neg), replace=False)
            anchor_labels[disable_index] = -1

        

        '''
        Labels have already been assigned to the anchors, now we need to
        assign locations to anchor boxes. Assign them the ground truth object
        with maximum IOU. 
        '''

        ## We have N valid achors, for each valid anchor, we have a corresponding 
        ## groundtruth bounding box. We compute them as below. For all anchors, 
        ## we need del_x, del_y, del_h, del_w, where (x,y) and (h,w) are center
        ## of the anchor and height, width respectively. 
        bbox = targets['boxes']
        max_iou_bbox = bbox[argmax_ious]

        ## Getting anchor centers and anchor dimensions(height, width)!
        anchor_height = inside_anchors[:,2] - inside_anchors[:,0] ## N x 1: height of all N anchors
        anchor_width = inside_anchors[:,3] - inside_anchors[:,1]  ## N x 1: width of all N anchors
        anchor_ctr_y = inside_anchors[:,0] + 0.5 * anchor_height      ## N x 1: y-coordinates of all Anchor centers
        anchor_ctr_x = inside_anchors[:,1] + 0.5 * anchor_width      ## N x 1: x-coordinates of all Anchor centers

        ## Getting ground truth BB centers and dimensions(height, width)

        ## N x 1: height of all N groundtruth bounding boxes for N anchors
        base_height = max_iou_bbox[:,2] - max_iou_bbox[:,0] 
        ## N x 1: height of all N groundtruth bounding boxes for N anchors
        base_width = max_iou_bbox[:,3] - max_iou_bbox[:,1]  
        ## N x 1: y-coordinates of the center of all N groundtruth bounding boxes for N anchors
        base_ctr_y = max_iou_bbox[:,0] + 0.5 * base_height      
        ## N x 1: x-coordinates of the center of all N groundtruth bounding boxes for N anchors
        base_ctr_x = max_iou_bbox[:,1] + 0.5 * base_width

        '''
        Using above information to find locations as required by Faster R-CNN
        '''

        eps = np.finfo(anchor_height.dtype).eps
        anchor_height = np.maximum(anchor_height, eps)
        anchor_width = np.maximum(anchor_width, eps)

        '''
        As required by Faster R-CNN
        '''
        dy = (base_ctr_y - anchor_ctr_y) / anchor_height
        dx = (base_ctr_x - anchor_ctr_x) / anchor_width
        dh = np.log(base_height / anchor_height)
        dw = np.log(base_width / anchor_width)

        anchor_locs = np.vstack((dy, dx, dh, dw)).transpose() ## Final locations of the anchors

        anchor_labels_final = np.empty((len(anchors), ), dtype = anchor_labels.dtype)
        anchor_labels_final.fill(-1)
        anchor_labels_final[inside_indices] = anchor_labels
        # print( np.sum( anchor_labels_final == 1) , np.sum( anchor_labels_final == 0))

        anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
        anchor_locations.fill(0)
        anchor_locations[inside_indices, :] = anchor_locs
        # print(orig_anchors[anchor_labels_final == 1])
        return anchor_locations, anchor_labels_final, orig_anchor