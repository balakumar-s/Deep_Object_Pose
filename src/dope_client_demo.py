#!/usr/bin/env python

#from __future__ import print_function
import sys
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from dope.srv import DetectObjPose
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
g_bridge = CvBridge()
g_img = None
import cv2

class dopeDetect(object):
    def __init__(self,init_node=False,topic_cam='/webcam/image_raw'):
        if(init_node):
            rospy.init_node('demo_dope_client')
        rospy.Subscriber(topic_cam, Image, self.__image_callback)
        self.got_image=False
        ### Basic functions
    def __image_callback(self,msg):
        self.g_img = msg#g_bridge.imgmsg_to_cv2(msg, "rgb8")
        self.got_image=True
    def detect_object(self,object_name):
        obj_pose=PoseStamped()
        while ( not self.got_image):
            continue
        _,obj_pose,_,_=self.detect_obj_client([object_name],self.g_img)
        return obj_pose
    def detect_obj_client(self,obj_names,rgb_img):
        rospy.wait_for_service('/dope/detect_obj_pose')
        try:
            srv = rospy.ServiceProxy('/dope/detect_obj_pose', DetectObjPose)
            resp1 = srv(obj_names,rgb_img,CameraInfo(),Image(),CameraInfo())
            return resp1.obj_idx,resp1.obj_pose, resp1.obj_rgb, resp1.success
        except rospy.ServiceException, e:
            print "Service call failed:"# %s",%e

if __name__=='__main__':
    topic_cam='/camera/rgb/image_raw'

    dope_detect=dopeDetect(True,topic_cam)

    print "available objects: mustard, cracker, gelatin, meat, soup, sugar,bleach, all"

    all_objects=['mustard', 'cracker', 'gelatin', 'meat', 'soup', 'sugar','bleach']
    if len(sys.argv) > 1:
        obj_name= [sys.argv[1]]
    else:
        obj_name= ['mustard']
    raw_input("Detect object?")
    while not rospy.is_shutdown():
        #raw_input("get new pose?")
        if dope_detect.got_image:
            if(obj_name[0] == 'all'):
                obj_idx,poses,rgb,succ=dope_detect.detect_obj_client(all_objects,dope_detect.g_img)

            else:
                obj_idx,poses,rgb,succ=dope_detect.detect_obj_client(obj_name,dope_detect.g_img)

            if(succ):
                print "Detected objects:"
                print obj_idx
                #print poses
                cv_img=g_bridge.imgmsg_to_cv2(rgb, "rgb8")
                cv2.imshow('image',cv_img)
                cv2.waitKey(2000) #ms
                
    cv2.destroyAllWindows() 
