'''
This file loads object detection annotations of bdd100k
and extracts dataset for desired classes. This way, we
can have subset of the dataset only for the classes we need.

Note: First, convert bdd100k annotations to coco format.(IMP) 
Load coco style annotations in this file to extract subset of
the desired dataset.

'''

import json
from pycocotools.coco import COCO

annotations_path = '/home/bdd_100k/bdd100k/coco_labels/bdd100k_labels_images_det_coco_train.json'

with open(annotations_path) as json_file: 
	data = json.load(json_file)  

'''
data variable is a dictionary and it has 4 keys, dict_keys(['categories', 'images', 'annotations', 'type']).

'''