#!/usr/bin/env python

#from __future__ import print_function
import sys
import rospy
from cv_bridge import CvBridge, CvBridgeError

from dope.srv import DetectObjPose
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
g_bridge = CvBridge()
g_img = None
import cv2

### Basic functions
def __image_callback(msg):
    '''Image callback'''
    global g_img
    g_img = msg#g_bridge.imgmsg_to_cv2(msg, "rgb8")


def detect_obj_client(obj_names,rgb_img):
    rospy.wait_for_service('/dope/detect_obj_pose')
    try:
        srv = rospy.ServiceProxy('/dope/detect_obj_pose', DetectObjPose)
        resp1 = srv(obj_names,rgb_img,CameraInfo(),Image(),CameraInfo())
        return resp1.obj_idx,resp1.obj_pose, resp1.obj_rgb, resp1.success
    except rospy.ServiceException, e:
        print "Service call failed:"# %s",%e

if __name__=='__main__':
    rospy.init_node('demo_dope_client')
    
    topic_cam='/camera/rgb/image_raw'

    print "available objects: mustard, cracker, gelatin, meat, soup, sugar, all"

    all_objects=['mustard', 'cracker', 'gelatin', 'meat', 'soup', 'sugar']
    if len(sys.argv) > 1:
        obj_name= [sys.argv[1]]
    else:
        obj_name= ['mustard']
    rospy.Subscriber(
        topic_cam, 
        Image, 
        __image_callback
    )
    raw_input("Detect object?")
    while not rospy.is_shutdown():
        #raw_input("get new pose?")
        if g_img is not None:
            if(obj_name[0] == 'all'):
                obj_idx,poses,rgb,succ=detect_obj_client(all_objects,g_img)

            else:
                obj_idx,poses,rgb,succ=detect_obj_client(obj_name,g_img)

            if(succ):
                print "Detected objects:"
                print obj_idx
                #print poses
                cv_img=g_bridge.imgmsg_to_cv2(rgb, "bgr8")
                cv2.imshow('image',cv_img)
                cv2.waitKey(2000) #ms
                
    cv2.destroyAllWindows() 
