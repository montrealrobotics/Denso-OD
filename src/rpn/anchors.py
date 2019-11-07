'''
Base class to generate anchors. 
This code will mostly be used for training!(You never know though!)
'''
import copy
import numpy as np
import torch
from torch import nn
from ..utils import Boxes
import math


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, device=torch.device('cuda')):
        super(AnchorGenerator, self).__init__()
        sizes         = cfg.ANCHORS.ASPECT_RATIOS
        aspect_ratios = cfg.ANCHORS.ANCHOR_SCALES
        self.num_images = cfg.TRAIN.BATCH_SIZE

        self.base_anchors = self.generate_base_anchors(sizes, aspect_ratios, device)
        self.register_buffer("cell_anchors", self.base_anchors)

    def _create_grid_offsets(self, size, stride, device):
        grid_height, grid_width = size
        shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(
            0, grid_height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def grid_anchors(self, grid_sizes, stride):
        anchors = []
       
        shift_x, shift_y = self._create_grid_offsets(grid_sizes, stride, self.base_anchors.device)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        anchors = shifts.view(-1, 1, 4) + self.base_anchors.view(1, -1, 4) # anchors: [H*W, num_of_achors_per_place=9, 4]
        anchors = anchors.reshape(-1, 4) # anchors: [tot_num_anchors=H*W*9, 4] : first 9 boxes are anchors for first (x,y) and so on. so

        return anchors

    def generate_base_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), device=torch.device('cuda')):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors, device=device).float()

    def forward(self, features_shape, stride):
        """
        Args:
            features (Tensor): Tensor of feature map on which to generate anchors.

        Returns:
            list[Boxes]: a list of #image elements.
        """
        # num_images = features_shape[0]
        grid_sizes = features_shape
        anchors = self.grid_anchors(grid_sizes, stride)
        anchors_image = Boxes(anchors) 

        anchors_batch = [copy.deepcopy(anchors_image) for _ in range(self.num_images)]
    
        # List(Boxes) len(anchor_batch) = batch_size, 
        # each element is a Box Strucutre which is list of all anchor in image i. 
        # Box = [HxWx9, 4]
        return anchors_batch 


class anchor_generator(object):
    """docstring for anchor_generator"""
    def __init__(self):
        super(anchor_generator, self).__init__()
        ## Do nothing!
    
    ## TODO: Automate the process of getting these inputs. Make the inputs configuration parameters.

    def get_anchors(self, image, feature_map, cfg):
        
        '''

        Input: 
        image: A single image(torch tensor) of size BZxCxHxW, BS = batchsize, C = channels, H = height, W = width
        feature_map: Torch tensor of size BSxFcxFhxFw, BS = batchsize, Fc = feature map channels, Fh = feature map height, Fw = feature map width
        aspect_ratios: An array of various anchor ratios
        anchor_scales: An array of various anchor scales

        Output: 
        Nx4 numpy array of anchor locations. N = number of anchors. 
        
        '''
        self.image = image
        self.feature_map = feature_map
        self.subsample = [int(round(self.image.size()[2]/self.feature_map.size()[2])),
                            int(round(self.image.size()[3]/self.feature_map.size()[3]))] ## Subsampling ratio. 

        # print(self.subsample)
        self.aspect_ratios = list(cfg.ANCHORS.ASPECT_RATIOS)
        self.anchor_scales = list(cfg.ANCHORS.ANCHOR_SCALES)

        ## Let's get anchor centres(wrt original image)
        return self.get_all_anchors(im_height = self.image.size()[2], 
                                    im_width = self.image.size()[3], 
                                    feature_height = self.feature_map.size()[2],
                                    feature_width = self.feature_map.size()[3],
                                    aspect_ratios = self.aspect_ratios,
                                    anchor_scales = self.anchor_scales)




    def get_valid_anchors(self):
        # self.anchor_data = self.get_all_anchors() ## TODO: Get all the necessary arguments
        valid_anchor_indices = np.where((self.anchor_data[:,0] >= 0) &
                             (self.anchor_data[:,1] >= 0) &
                             (self.anchor_data[:,2] <= im_height) &
                             (self.anchor_data[:,3] <= im_width))[0]

        # print(valid_anchor_indices.shape)


        ## Constructing an array holding valid anchor boxes
        self.valid_anchor_boxes = self.anchor_data[valid_anchor_indices]

    def generate_anchors(self, anchor_center, aspect_ratios, anchor_scales):
    
        '''
        Input: 
        
        anchor_center: [y, x] ## Notation similar to how BB are defined. Co-ordinates in original image.
        aspect_ratios: [ar1, ar2, ar3]
        anchor_scales: [as1, as2, as3]
        
        Output: A numpy array of 9x4, generates 9 anchors with center as given anchor center

        Testing the function:

        anchor_center = [100, 100]
        aspect_ratios = [0.5, 1, 2]
        anchor_scales = [128, 256, 512]
        anchor_base = generate_anchors(anchor_center, aspect_ratios, anchor_scales, sub_sample)

        '''
        
        ## We will have 9 anchors, each with 4 co-ordinates
        anchor_base = np.zeros((len(anchor_scales) * len(aspect_ratios), 4), dtype=np.float32)
        
        ## Determining co-ordinates for all the anchors
        for i in range(len(anchor_scales)):
            for j in range(len(aspect_ratios)):
           
                ## height and width should be such that h/w = aspect_ratios[j]
                anchor_width = anchor_scales[i]*np.sqrt(aspect_ratios[j])
                anchor_height = anchor_scales[i]*np.sqrt(1.0/aspect_ratios[j])
                # print("Anchor height and anchor widths are: ", anchor_height, anchor_width)
                anchor_index = i*len(anchor_scales) + j 
                
                ## y co-ordinate of top left corner
                anchor_base[anchor_index, 0] = anchor_center[0] - anchor_height/2.0
                ## x co-ordinate of top left corner
                anchor_base[anchor_index, 1] = anchor_center[1] - anchor_width/2.0
                
                ## y co-ordinate of bottom right corner
                anchor_base[anchor_index, 2] = anchor_center[0] + anchor_height/2.0
                ## x co-ordinate of bottom right corner
                anchor_base[anchor_index, 3] = anchor_center[1] + anchor_width/2.0
                
        # print("Anchor base!!!!", anchor_base)     
        return anchor_base

    ## DONE: Automate the process of getting these inputs. Make the inputs configuration parameters.

    def get_anchor_centers(self, im_height, im_width, feature_height, feature_width):
    
        '''
        Given image dimensions and subsampling ratio from backbone, we get all anchor centers with respect to original image.
        
        Inputs: 
        im_height: image height
        im_width: image width
        feature_height: feature_height
        feature_width: feature_width
        
        Outputs:
        ctr_y, ctr_x: center points
        
        Testing the function:
        ctr_y, ctr_x = get_anchor_centers(im_height = 800, im_width = 450, sub_sample = 16)
        
        '''
        ## Here x is facing towards the right, y is facing downwards
        # print(im_height, im_width, sub_sample)
        sub_sample = [int(round(self.image.size()[2]/self.feature_map.size()[2])),
                            int(round(self.image.size()[3]/self.feature_map.size()[3]))] ## Subsampling ratio.

        ## let's get anchor centers!
        ctr_y = np.arange(sub_sample[0]//2, (feature_height)*sub_sample[0], sub_sample[0])
        ctr_x = np.arange(sub_sample[1]//2, (feature_width)*sub_sample[1], sub_sample[1])
        # print("shapes", ctr_x.shape, ctr_y.shape)
        return ctr_y, ctr_x

    def get_all_anchors(self,im_height, im_width, feature_height, feature_width, aspect_ratios, anchor_scales):
    
        '''
        Inputs: 
        
        im_height: image height
        im_width: image width
        sub_sample: subsampling ratio of convolutional feature extractor
        aspect_ratios: [ar1, ar2, ar3]
        anchor_scales: [as1, as2, as3]
        
        Output: 
        A numpy array
        
        '''
        
        ## getting anchor centers with respect to original image
        ctr_y, ctr_x = self.get_anchor_centers(im_height=im_height, im_width = im_width, feature_height = feature_height, feature_width = feature_width)

        
        ## At each pixel, we will have 9 anchors, total number of pixels in convolutional 
        ## feature map are (im_height*im_width)//(sub_sample*sub_sample). Hence,
        ## total number of anchors are,
        num_of_pixels = feature_height*feature_width
        anchors_per_pixel = len(aspect_ratios)*len(anchor_scales)
        
        ## Each anchor has 4 points(y1, x1, y2, x2)
        ## This variable holds information of all the anchors at all the pixel locations
        ## defined in the original image
        anchor_data = np.zeros((num_of_pixels, anchors_per_pixel, 4), dtype=np.float32)
        
        
        ## Let's iterate through all anchor centers and get their corresponding anchor co-ordinates
        # Dishank: Double for loop is slow
        index = 0
        for i in range(len(ctr_y)):
            for j in range(len(ctr_x)):
                
                anchor_center = [ctr_y[i], ctr_x[j]]
                
                ## generate all 9 anchors for this anchor center, output 9x4 array
                anchor_base = self.generate_anchors(anchor_center, aspect_ratios, anchor_scales)
                
                ## Populating anchor data matrix with appropriate co-ordinates
                anchor_data[index,:,:] = anchor_base
                
                index = index + 1
        
        anchor_data = anchor_data.reshape(num_of_pixels*anchors_per_pixel, 4)
        
        return anchor_data  ## Shape: Total number of anchorsx4

