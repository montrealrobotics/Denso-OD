'''
Let's check the preprocessing.
'''

import torch
import sys
import numpy as np

## Inserting path of src directory
sys.path.insert(1, '../..')
from src.config import Cfg as cfg
from src.backbone import Backbone
from src.preprocess import image_transform ## It's a function, not a class. 

import matplotlib.image as mpimg ## To load the image

img = mpimg.imread('test.jpg')		## Gives RGB image of dimension H x W x C with inten values between 0-255

transform = image_transform(cfg)
img_new = transform(img)
img_new = torch.unsqueeze(img_new, dim=0)
print(img_new.shape)