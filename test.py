#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
import numpy as np
import cv2


def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    np_arr = np.fromstring(data.data, np.uint8)
    color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    print(color_image.shape)
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('listenerasasdsadas', anonymous=True)

    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, callback,queue_size = 1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
