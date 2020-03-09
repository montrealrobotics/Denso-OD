#!/usr/bin/env python3

import sys
import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

import rospy
import cv2

print(sys.path)

from src.utils import utils
from src.eval.detection_map import DetectionMAP
from src.config import Cfg as cfg
from src.tracker.track import MultiObjTracker
from src.utils import Instances
from src.utils import Boxes
from src.utils import utils

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from denso.msg import BoundingBox2D, Classification2D, Variance2D, Instances as Instances_msg

from model import create_model



class model_inference(object):
    """docstring for model_inference."""

    def __init__(self):
        super(model_inference, self).__init__()

        checkpoint_path = rospy.get_param('/model_weight')
        rospy.loginfo("Model weights path: %s", checkpoint_path)

        self.model = create_model(checkpoint_path)
        self.tracker = MultiObjTracker(max_age=4)
        self.is_training = False
        self.cv_bridge = CvBridge()
        self.img_transform = utils.image_transform(cfg)

        self.output_image_pub = rospy.Publisher("/output_img", Image, queue_size=10)
        self.detected_boxes = rospy.Publisher("/detected_boxes", Instances_msg, queue_size=10)

        rospy.loginfo("Model is built and ready to be used")


    def single_img_inference(self, input_img):
        # Converting ros image to opencv image
        path = input_img.header.seq 
        in_image = self.cv_bridge.imgmsg_to_cv2(input_img, desired_encoding="passthrough") # our incoming encoding is rgb8, hence we don't need to change
        # cv2.imshow('img',in_image)
        # cv2.waitKey(0)

        in_image = self.img_transform(in_image)
        # Input to the model is list of images.
        in_image = in_image.unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            _, instances, _, _ = self.model(in_image, is_training=self.is_training)

            instances[0].toList()

            for instance in instances:
                self.tracker.predict()
                self.tracker.update(instance.pred_boxes, instance.pred_variance)

            updated_instances = Instances((1242,375), pred_boxes=Boxes(torch.tensor([x.mean[:4] for x in self.tracker.tracks])), pred_variance=torch.tensor([x.get_diag_var()[:4] for x in self.tracker.tracks]))
            updated_instances.toList()
            in_image = torch.squeeze(in_image, 0)
            output_img = utils.single_disk_logger(in_image, updated_instances, None, image_path=path)

        # for img in output_imgs:
            ros_img = self.cv_bridge.cv2_to_imgmsg(output_img, encoding="rgb8")

            instances = Instances_msg()
            instances.detections = [BoundingBox2D(*x) for x in updated_instances.pred_boxes]
            instances.variances = [Variance2D(*x) for x in updated_instances.pred_variance]

            self.output_image_pub.publish(ros_img)
            self.detected_boxes.publish(instances)




def main(args):

	rospy.init_node('model')
	model_infer = model_inference()
	sub = rospy.Subscriber("/image_raw", Image, model_infer.single_img_inference)
	rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
