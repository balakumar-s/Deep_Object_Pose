#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function
import yaml
import sys 

import numpy as np
import cv2

import rospy
import rospkg
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageSensor_msg
from geometry_msgs.msg import PoseStamped

from dope.srv import *

from PIL import Image
from PIL import ImageDraw

# Import DOPE code
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')
sys.path.append("{}/src/inference".format(g_path2package))
from cuboid import *
from detector import *



class dopeClass(object):

    def __init__(self):
        self.g_bridge = CvBridge()
        self.g_img = None
        self.g_draw = None
        rospy.init_node('dope_service', anonymous=True)
        self.config_detect = None
        self.models = None
        self.draw_colors = None
        self.pnp_solvers = None
        self.srv = None
        self.dope_init=False
    ### Basic functions
    def __image_callback(self,sensor_msg):
        g_img = self.g_bridge.imgmsg_to_cv2(sensor_msg, "rgb8")
        # cv2.imwrite('img.png', cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))  # for debugging
        return g_img

    ### Code to visualize the neural network output

    def DrawLine(self,point1, point2, lineColor, lineWidth):
        '''Draws line on image'''
        if not point1 is None and point2 is not None:
            self.g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

    def DrawDot(self,point, pointColor, pointRadius):
        '''Draws dot (filled circle) on image'''
        if point is not None:
            xy = [
                point[0]-pointRadius, 
                point[1]-pointRadius, 
                point[0]+pointRadius, 
                point[1]+pointRadius
            ]
            self.g_draw.ellipse(xy, 
                           fill=pointColor, 
                           outline=pointColor
            )

    def DrawCube(self,points, color=(255, 0, 0)):
        '''
        Draws cube with a thick solid line across 
        the front top edge and an X on the top face.
        '''
        
        lineWidthForDrawing = 2
        
        # draw front
        self.DrawLine(points[0], points[1], color, lineWidthForDrawing)
        self.DrawLine(points[1], points[2], color, lineWidthForDrawing)
        self.DrawLine(points[3], points[2], color, lineWidthForDrawing)
        self.DrawLine(points[3], points[0], color, lineWidthForDrawing)
        
        # draw back
        self.DrawLine(points[4], points[5], color, lineWidthForDrawing)
        self.DrawLine(points[6], points[5], color, lineWidthForDrawing)
        self.DrawLine(points[6], points[7], color, lineWidthForDrawing)
        self.DrawLine(points[4], points[7], color, lineWidthForDrawing)
        
        # draw sides
        self.DrawLine(points[0], points[4], color, lineWidthForDrawing)
        self.DrawLine(points[7], points[3], color, lineWidthForDrawing)
        self.DrawLine(points[5], points[1], color, lineWidthForDrawing)
        self.DrawLine(points[2], points[6], color, lineWidthForDrawing)
        
        # draw dots
        self.DrawDot(points[0], pointColor=color, pointRadius = 4)
        self.DrawDot(points[1], pointColor=color, pointRadius = 4)
        
        # draw x on the top 
        self.DrawLine(points[0], points[5], color, lineWidthForDrawing)
        self.DrawLine(points[1], points[4], color, lineWidthForDrawing)


    def detect_obj(self,obj_names,rgb_info,rgb_msg):
        
        # Copy and draw image
        self.g_img=self.__image_callback(rgb_msg)
        img_copy = self.g_img.copy()
        im = Image.fromarray(img_copy)
        
        self.g_draw = ImageDraw.Draw(im)

        detected_poses=[]
        obj_idx=[]
        draw_img=None
        for obj_name in obj_names:
            if(obj_name in self.models.keys()):
            
                # Detect object
                results = ObjectDetector.detect_object_in_image(
                    self.models[obj_name].net, 
                    self.pnp_solvers[obj_name],
                    self.g_img,
                    self.config_detect
                )
                
                # Publish pose and overlay cube on image
                for i_r, result in enumerate(results):
                    if result["location"] is None:
                        continue
                    loc = result["location"]
                    ori = result["quaternion"]                    
                    msg = PoseStamped()
                    msg.header.frame_id = rgb_msg.header.frame_id #params["frame_id"]
                    msg.header.stamp = rospy.Time.now()
                    CONVERT_SCALE_CM_TO_METERS = 100
                    msg.pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.orientation.x = ori[0]
                    msg.pose.orientation.y = ori[1]
                    msg.pose.orientation.z = ori[2]
                    msg.pose.orientation.w = ori[3]
                    detected_poses.append(msg)
                    obj_idx.append(obj_name)
                    # Publish
                    #pubs[m].publish(msg)
                    #pub_dimension[m].publish(str(params['dimensions'][m]))
                    
                    # Draw the cube
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))
                        self.DrawCube(points2d, self.draw_colors[obj_name])
                
                        # Publish the image with results overlaid
                        draw_img=CvBridge().cv2_to_imgmsg(
                            np.array(im)[...,::-1], 
                            "rgb8")
            else:
                print ("object: ",obj_name," not in trained models!!")
        return obj_idx,detected_poses,draw_img

    def init_weights(self,params):
        
        pubs = {}
        models = {}
        pnp_solvers = {}
        pub_dimension = {}
        draw_colors = {}
        dist_coeffs = np.zeros((4,1))
        
        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = params['thresh_angle']
        self.config_detect.thresh_map = params['thresh_map']
        self.config_detect.sigma = params['sigma']
        self.config_detect.thresh_points = params["thresh_points"]
        # Initialize parameters
        matrix_camera = np.zeros((3,3))
        matrix_camera[0,0] = params["camera_settings"]['fx']
        matrix_camera[1,1] = params["camera_settings"]['fy']
        matrix_camera[0,2] = params["camera_settings"]['cx']
        matrix_camera[1,2] = params["camera_settings"]['cy']
        matrix_camera[2,2] = 1
        draw_img=None
        if "dist_coeffs" in params["camera_settings"]:
            dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
            

        # For each object to detect, load network model, create PNP solver, and start ROS publishers
        for model in params['weights']:
            models[model] =\
                            ModelData(
                                model, 
                                g_path2package + "/weights/" + params['weights'][model]
                            )
            models[model].load_net_model()
            
            draw_colors[model] = \
                                 tuple(params["draw_colors"][model])
            pnp_solvers[model] = \
                                 CuboidPNPSolver(
                                     model,
                                     matrix_camera,
                                     Cuboid3d(params['dimensions'][model]),
                                     dist_coeffs=dist_coeffs
                                 )
            '''
            pubs[model] = \
                          rospy.Publisher(
                              '{}/pose_{}'.format(params['topic_publishing'], model), 
                              PoseStamped, 
                              queue_size=10
                          )
            pub_dimension[model] = \
                                   rospy.Publisher(
                                       '{}/dimension_{}'.format(params['topic_publishing'], model),
                                       String, 
                                       queue_size=10
                                   )
            
            # Start ROS publisher
            pub_rgb_dope_points = \
                                  rospy.Publisher(
                                      params['topic_publishing']+"/rgb_points", 
                                      ImageSensor_msg, 
                                      queue_size=10
                                  )
    
            # Starts ROS listener
            rospy.Subscriber(
                topic_cam, 
                ImageSensor_msg, 
                __image_callback
            )
            '''
        self.models=models
        self.draw_colors=draw_colors
        self.pnp_solvers=pnp_solvers
        print ('Initialized dope weights')
        self.dope_init=True
        
    def srv_call(self,req):
        if(self.dope_init):
            obj_idx,poses,obj_rgb=self.detect_obj(req.obj_names,req.rgb_info,req.rgb_image)
            if(len(poses)==0):
                return DetectObjPoseResponse(obj_idx,poses,obj_rgb,False)
            else:
                return DetectObjPoseResponse(obj_idx,poses,obj_rgb,True)
        else:
            return DetectObjPoseResponse([],[],Image(),False)
    def run_service(self):
        loop_rate=rospy.Rate(5)
        # create service
        self.srv = rospy.Service('/dope/detect_obj_pose', DetectObjPose, self.srv_call)

        while not rospy.is_shutdown():
            
            loop_rate.sleep()
        


if __name__ == "__main__":
    '''Main routine to run DOPE'''
            
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "config_pose.yaml"
    rospack = rospkg.RosPack()
    params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)

    #topic_cam = params['topic_camera']

    
    try :
        # initialize dope class
        dope=dopeClass()
        # load weights
        dope.init_weights(params)
    except rospy.ROSInterruptException:
        pass

    # run loop for service
    dope.run_service()
