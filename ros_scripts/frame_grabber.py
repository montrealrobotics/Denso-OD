

from __future__ import print_function

import sys
import os
from os import listdir
from os.path import isfile, join

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_folder_publisher:
    def __init__(self):
        self.cv_bridge = CvBridge()

        self.topic_name = rospy.get_param('~topic_name', '/image_raw')
        rospy.loginfo("Publishing Images to topic  %s", self._topic_name)

        self.image_publisher = rospy.Publisher(self._topic_name, Image, queue_size=10)

        self.rate = rospy.get_param('~publish_rate', 15)
        rospy.loginfo("Publish rate set to %s hz", self.rate)

        self.sort_files = rospy.get_param('~sort_files', True)
        # rospy.loginfo("[%s] (sort_files) Sort Files: %r", self.__app_name, self._sort_files)

        self.frame_id = rospy.get_param('~frame_id', 'camera')
        rospy.loginfo("Frame ID set to  %s", self.frame_id)

        # self.loop = rospy.get_param('~loop', 1)
        # rospy.loginfo("[%s] (loop) Loop  %d time(s) (set it -1 for infinite)", self.__app_name, self._loop)

        self.image_folder = rospy.get_param('~image_folder',
                                    '/network/home/bansaldi/Denso-OD/datasets/kitti_tracking/images/training/0001')

        if self.image_folder == '' or not os.path.exists(self.image_folder) or not os.path.isdir(self.image_folder):
            rospy.logfatal("Invalid Image folder")
            sys.exit(0)
        rospy.loginfo("Reading images from %s", self.image_folder)

    def run(self):

        ros_rate = rospy.Rate(self.rate)

        files_in_dir = [f for f in listdir(self.image_folder) if isfile(join(self.image_folder, f))]
        if self.sort_files:
            files_in_dir.sort()
        try:
            while self.loop != 0:
                for f in files_in_dir:
                    if not rospy.is_shutdown():
                        # if isfile(join(self.image_folder, f)):
                        cv_image = cv2.imread(join(self.image_folder, f))
                        if cv_image is not None:
                            ros_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "rgb8")
                            ros_msg.header.seq = join(self.image_publisher, f)
                            ros_msg.header.frame_id = self.frame_id
                            ros_msg.header.stamp = rospy.Time.now()
                            self.image_publisher.publish(ros_msg)
                            rospy.loginfo("[%s] Published %s", self.__app_name, join(self.image_folder, f))
                        else:
                            rospy.loginfo("[%s] Invalid image file %s", self.__app_name, join(self.image_folder, f))
                        ros_rate.sleep()
                    else:
                        return
                self.loop = self.loop - 1
        except CvBridgeError as e:
            rospy.logerr(e)


def main(args):
    rospy.init_node('raw_image_publisher')

    image_publisher = folder_to_imgSeq()
    image_publisher.run()


if __name__ == '__main__':
    main(sys.argv)
