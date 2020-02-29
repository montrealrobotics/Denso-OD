from __future__ import print_function

import sys
import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import time

import rospy
import cv2

from ..utils import utils
from ..eval.detection_map import DetectionMAP
import matplotlib.pyplot as plt
from src.config import Cfg as cfg
from ..tracker.track import MultiObjTracker
from ..utils import Instances
from ..utils import Boxes


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from denso-od.msg import Instances

from .model import create_model



class model_inference(object):
    """docstring for model_inference."""

    def __init__(self, arg):
        super(model_inference, self).__init__()

        checkpoint_path = rospy.get_param('~model_weights')

        self.model = create_model(checkpoint_path)
        self.tracker = MultiObjTracker(max_age=4)
        self.is_training = False
        self.cv_bridge = CvBridge()


    def single_img_inference(input_img):
        # Converting ros image to opencv image
        in_image = self.cv_bridge.imgmsg_to_cv2(input_img, desired_encoding="rgb8")
        in_image = transform(in_image)

        # Input to the model is list of images.
        in_image = [in_image]
        with torch.no_grad():
            _, instances, _, _ = self.model(in_images, is_training=self.is_training)

			instances[0].toList()

			for instance in instances:
				self.tracker.predict()
				self.tracker.update(instance.pred_boxes, instance.pred_variance)

			updated_instances = [Instances((1242,375), pred_boxes=Boxes(torch.tensor([x.mean[:4] for x in tracker.tracks])), pred_variance=torch.tensor([x.get_diag_var()[:4] for x in tracker.tracks]))]
            updated_instances[0].toList()
            output_imgs = utils.disk_logger(in_images, results_dir+"/tracked", updated_instances, rpn_proposals, img_paths)
            updated_instances = updated_instances[0]

        for img in output_imgs:
            ros_img = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")

            instances = Instances()
            instances.detections = updated_instances.pred_boxes
            instances.variances = updated_instances.pred_variance

            output_image_pub.publish(ros_img)
            detected_boxes.pub(instances)




def main(args):
    rospy.init_node('model')
    model_infer = model_inference()

    output_image_pub = rospy.Publisher("/ouput_img", Image, queue_size=10)
    detected_boxes = rospy.Publisher("/detected_boxes", custom_box_msg_type, queue_size=10)
    sub = rospy.Subcriber("/raw_image", Image, model_infer.single_img_inference)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
