#!/usr/bin/env python3

from __future__ import print_function

import sys
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
import pykitti

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from numpy_pc2 import array_to_xyzi_pointcloud2f


class Kitti_Publisher():
    def __init__(self, data):

        self.images = data.cam2
        self.velodyne_scans = data.velo

        self.cv_bridge = CvBridge()

        self.image_topic = rospy.get_param('~image_topic', '/image_raw')
        rospy.loginfo("Publishing Images to topic  %s", self.image_topic)

        self.velodyne_topic = rospy.get_param('~velodyne_topic', '/velodyne')
        rospy.loginfo("Publishing Images to topic  %s", self.velodyne_topic)

        self.image_publisher = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.velodyne_publisher = rospy.Publisher(self.velodyne_topic, PointCloud2, queue_size=10)

        self.rate = rospy.get_param('~publish_rate', 5)
        rospy.loginfo("Publish rate set to %s hz", self.rate)

        # self.sort_files = rospy.get_param('~sort_files', True)
        # rospy.loginfo("[%s] (sort_files) Sort Files: %r", self.__app_name, self._sort_files)

        self.cam_frame_id = rospy.get_param('~frame_id_cam', 'cam2')
        rospy.loginfo("Camera Frame ID set to  %s", self.cam_frame_id)

        self.velo_frame_id = rospy.get_param('~frame_id_velo', 'velodyne')
        rospy.loginfo("Velodyne Frame ID set to  %s", self.velo_frame_id)

        self.loop = rospy.get_param('~loop', 1)
        # rospy.loginfo("[%s] (loop) Loop  %d time(s) (set it -1 for infinite)", self.__app_name, self._loop)

    def run(self):

        ros_rate = rospy.Rate(self.rate)

        while self.loop != 0:
            for img, velo_scan in zip(self.images, self.velodyne_scans):
                if not rospy.is_shutdown():
                    cv_image = np.array(img) # Converting PIL image to cv2
                    pc2 = array_to_xyzi_pointcloud2f(velo_scan, frame_id = self.velo_frame_id)

                    # ros_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8") # bgr8 encoding gives error in image_view, hence moved to rgb8 and chaged the order above as well. 
                    ros_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "rgb8") # This encoding just tells what is the encoding of the image, here the fucntion does not change the encoding of image to the mentioned
                    # ros_msg.header.seq = join(self.image_folder, f)
                    ros_msg.header.frame_id = self.cam_frame_id
                    ros_msg.header.stamp = rospy.Time.now()
    
                    self.image_publisher.publish(ros_msg)
                    self.velodyne_publisher.publish(pc2)
    
                    # rospy.loginfo("Published %s", f)
    
                    ros_rate.sleep()

                else:
                    return

            self.loop = self.loop - 1


def main(args):
    rospy.init_node('kitti_publisher')

    base_folder = rospy.get_param('~velodyne_folder',
                                '/home/dishank/denso-ws/src/denso/datasets/kitti_tracking/training')

    if base_folder == '' or not os.path.exists(base_folder) or not os.path.isdir(base_folder):
        rospy.logfatal("Invalid Image folder")
        sys.exit(0)
    rospy.loginfo("Reading images from %s", base_folder)
    
    sequence = rospy.get_param('~sequence', '0001')

    tracker_data = pykitti.tracking(base_folder, sequence)

    kitti_publisher = Kitti_Publisher(tracker_data)
    kitti_publisher.run()


if __name__ == '__main__':
    main(sys.argv)
