#!/usr/bin/env python

from turtle import right
from matplotlib import axes
from geometry_msgs.msg import Twist
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import time
import imutils
import random
import io
from image_utils import contour,get_gray,draw_cricle,hsv_threshold
from scipy.spatial.distance import cosine as cosine_distance

#controller node name
controller_node = "test_controller"
#competition ros topics
camera_feed_topic = "/R1/pi_camera/image_raw"
cmd_vel_topic = "/R1/cmd_vel"
license_plate_topic = "/license_plate"

#internal debug publisher topics
debug_roadimg_topic = "/test_controller/road_image"
debug_stopimg_topic = "/test_controller/crosswalk_image"
debug_errplot_topic = "/test_controller/error_plot"

#debug /tune mode
DEBUG = True

#pid driving controller parameters

DRIVE_SPEED = 0.3

RIGHT_BORDER_MODIFIER = 325

K_P = 8#3.5
K_D = 1#2.5
K_I = 0#3

FREQUENCY = 1000 # #hz
MAX_ERROR = 20
G_MULTIPLIER = 0.05
ERROR_HISTORY_LENGTH = 5 #array size

#stopping parameters
STOP_PROXIMITY_THRESHOLD =550
STOP_DURATION = 2 #seconds



#main method that is executed when we rosrun or roslaunch 
def main():
    rospy.init_node(controller_node)
    controller = PID_controller()
    rate = rospy.Rate(FREQUENCY)


    start_time = rospy.get_time()
    timer_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    timer_pub.publish(String("Team1,pass,0,AAAA"))
    end_time = start_time + 60*500 #TODO CHANGE THIS FOR TIME

    while not rospy.is_shutdown() and rospy.get_time() < end_time:
        controller.control_loop()
        rate.sleep()

    timer_pub.publish(String("Team1,pass,-1,AAAA"))


    controller.send_stop_cmd()

def get_plot_img(error_array, time_array):
    if(error_array is not None and time_array is not None):
        plt.clf()
        fig, ax=plt.subplots()
        ax.set_ylim((-1*MAX_ERROR,MAX_ERROR))
        ax.plot(time_array[:-1],error_array[:-1])
        plt.pause(0.0000001)

        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    
    return None

class CameraFeed:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_feed_topic, Image, self.callback, queue_size=1)
        self.latest_img = Image()
        self.empty = True
        self.frameCount = 0
        self.img_area = 0

    #subscriber callback that receives latest image from camera feed
    def callback(self, img):
        try:
            self.frameCount = self.frameCount + 1
            self.latest_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            if(self.img_area == 0):
                self.img_area = self.latest_img.shape[0]*self.latest_img.shape[1]
            self.empty = False
            # self.save_image()
        except CvBridgeError as e:
            print(e)

    def save_image(self):
        cv2.imwrite('imgs/img_'+str(self.frameCount)+'.jpg',self.latest_img)
        print('SAVED IMAGE:','img_',self.frameCount)

class DriveState(Enum):
    DRIVE_FORWARD = 1
    TURN = 2
    STOPPING = 3
    STOPPED = 4
    CROSSING = 5
    FINISHED_CROSSING = 6
    READ_PARKING = 7

class PID_controller():
    def __init__(self):
        self.road_clr = [85,85,85]
        self.border_clr = [0,0,0]
        self.stop_clr  = [0,0,255]
        
        #init drive command subscriber
        self.drive_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)

        #init camera feed
        self.camera_feed = CameraFeed()

        #debug topics
        self.debug_errorimg_pub = rospy.Publisher(debug_errplot_topic, Image, queue_size=10)
        self.debug_roadimg_pub = rospy.Publisher(debug_roadimg_topic, Image, queue_size=10)
        self.debug_stopimg_pub = rospy.Publisher(debug_stopimg_topic, Image, queue_size=10)


        self.last_error = 0
        self.last_time = rospy.get_time()

        self.error_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5
        self.time_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5

        self.index = 0

        self.move_state = 0
        self.last_stop_time = 0
        self.pedestrian_crossing = False
        self.last_stop_points = []

        #Setting an arbitrary state to start timer at
        self.state = 1
        self.drive_state = DriveState.DRIVE_FORWARD

        self.debug_items = []


    #main control loop for pid controller
    def control_loop(self):
        self.state+=1

        if(self.camera_feed.empty is False):
            image = self.camera_feed.latest_img
            move = Twist()

            # error = self.get_error_road(image)
            # error_border = self.get_error_border(image)
            # error = (error + error_border)/2
            error = self.get_error_right(image)
            g = self.calculate_pid(error)

            stop_flag = self.detect_stop(image)
            if(stop_flag == 1):
                self.drive_state = DriveState.STOPPED
            if(stop_flag == 2):
                self.drive_state = DriveState.CROSSING
            if(stop_flag ==3):
                self.drive_state == DriveState.DRIVE_FORWARD
            

        
            if(self.drive_state == DriveState.DRIVE_FORWARD or self.drive_state == DriveState.CROSSING):
                move = self.get_move_cmd(g)
            
            # if(self.state < 100):
            #     move.angular.z = 0.3

            self.camera_feed.empty = True
            self.publish_debug()

            #publish move command to robot
            self.drive_pub.publish(move)

    def publish_debug(self):
        """
        @brief: publish debug items to ros debug topics
        """
        while len(self.debug_items) > 0:
            d = self.debug_items.pop()
            if(d.road_img is not None):
                self.debug_roadimg_pub.publish(self.camera_feed.bridge.cv2_to_imgmsg(d.road_img,"bgr8"))    
            if(d.stop_img is not None):
                self.debug_stopimg_pub.publish(self.camera_feed.bridge.cv2_to_imgmsg(d.stop_img,"bgr8"))
            if(d.err_img is not None):
                self.debug_errorimg_pub.publish(self.camera_feed.bridge.cv2_to_imgmsg(d.err_img,"bgr8"))

    def get_move_cmd(self,g,x_speed=DRIVE_SPEED):
        """
        get move command based on p+i+d= g
        g only impacts angular.z
        for now linear.x is constant
        """
        move = Twist()

        move.angular.z = g * G_MULTIPLIER
        move.linear.x =  x_speed #np.interp(np.abs(move.angular.z), [1,0], LINEAR_SPEED_RANGE)

        print('g', g)
        print('z', move.angular.z)
        print('x', move.linear.x)

        return move

    def send_stop_cmd(self):
        move = Twist()
        move.angular.z = 0
        move.linear.x = 0
        self.drive_pub.publish(move)

    def calculate_pid(self, error):
        """
        calculate pid : proportionality, integration and derivative values
        tries to do a better job by using numpy trapezoidal integration as well as numpy gradient
        uses error arrays for integration
        error arrays are of size error_history 9global variable)
        """
        curr_time = rospy.get_time()
        self.error_array[self.index] = error
        self.time_array[self.index] = curr_time

        # derivative
        derivative = np.gradient(self.error_array, self.time_array)[self.index]

        # trapezoidal integration
        integral = np.trapz(self.error_array, self.time_array)

        p = K_P * error
        i = K_I * integral
        d = K_D * derivative

        if (self.index == self.error_array.size - 1):
            self.error_array = np.roll(self.error_array, -1)
            self.time_array = np.roll(self.time_array, -1)
        else:
            self.index += 1

        # debug_item = RoadDebugItem(error_array=self.error_array,time_array=self.time_array)
        # self.debug_items.append(debug_item)

        print("pid ", p, i, d)
        g = p + i + d
        return g
    
    def get_error_border(self, image):
        """
        obtains error from camera feed frame by using cv2.findContours

        image is an np array, some useful techniques are applied before contour detection 
        to make it less prone to error

        contour basically outlines the pathes/borders on either side of the road
        from there the centre of the road to obtained relative to the 2 paths

        The error is the displacement of the robot/camera from the actual path centre
        this displacement is mapped to a smaller value with a maximum of max_error using np.interpolate
        the mapped displacement is returned as the error to be used for pid calculations
        """
        xerror = 0
        img_debug = image.copy()

        max_reading_error = image.shape[1] / 2
        min_reading_error = 25

        # centre points relative to robot camera (i.e. centre of image)
        pt_robot = np.array([image.shape[1],image.shape[0]])/2

        #contour detection with white filter
        gray = get_gray(image,clr='w')
        contours,pts = contour(gray,limit=3,apply_gray=False,pre_process=True,inv_thresh=255)

        for p in pts:
            pt = (p[0],p[1])
            draw_cricle(img_debug,pt,clr=(0,0,255))
            d = RoadDebugItem(road_img=img_debug)
            self.debug_items.append(d)

        pt_path = np.array([0,0])
        #check that more than one contour is detected
        if(len(pts) > 1):
            pt_path = pts[0] + (pts[1] - pts[0])/2

            displacement = pt_robot - pt_path

            sign_x = displacement[0] / np.abs(displacement[0])

            xerror = sign_x * np.interp(np.abs(displacement[0]),
                                        [min_reading_error, max_reading_error],
                                        [0, MAX_ERROR])
            

        self.last_error = xerror
        print('X Error: ', xerror)

        return xerror

    def detect_stop(self,image):
        detect_flag = 0
        height_lim = image.shape[0] * 0.95

        red_mask = hsv_threshold(image,clr='r')
        _,pts_red = contour(red_mask,limit=3,apply_gray=True,pre_process=False,area_limit=0.2)
        for pt in pts_red:
            draw_cricle(red_mask,(pt[0],pt[1]),clr=(0,255,0))

        if(len(pts_red) ==2):
            if(pts_red[0][1] > height_lim):
                if(self.drive_state==DriveState.DRIVE_FORWARD):
                    detect_flag = 1
                    pt = pts_red[0]
                    msg = 'STOP DETECTED ' 
                    cv2.putText(red_mask, msg, (pt[0], pt[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                elif self.drive_state == DriveState.CROSSING:
                    detect_flag = 3
                    pt = pts_red[0]
                    msg = 'FINISHED CROSSSING ' 
                    cv2.putText(red_mask, msg, (pt[0], pt[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            elif(pts_red[0][1] < height_lim and pts_red[1][1] < height_lim):
                detect_flag = 2
                msg = 'PEDESTRIAN CROSSED ' 
                cv2.putText(red_mask, msg, (pt[0], pt[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        elif len(pts_red) == 1:
            if (pts_red[0][1] < height_lim and self.drive_state == DriveState.STOPPED):
                detect_flag = 1
                msg = 'PEDESTRIAN CROSSED ' 
                cv2.putText(red_mask, msg, (pt[0], pt[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        d = RoadDebugItem(stop_img=red_mask)
        self.debug_items.append(d)
        
        self.last_stop_points = pts_red
        return detect_flag

    def get_error_right(self,image):
        img_debug = image.copy()
        x_error = 0
        max_reading_error = image.shape[1] / 2
        min_reading_error = 25
        detect = False

        pt_robot = np.array([image.shape[1],image.shape[0]])/2

        gray = get_gray(image,clr='w')
        _,pts_white = contour(gray,limit=3,apply_gray=False,pre_process=True,inv_thresh=255)

        if(len(pts_white) > 0 ):
            detect = True
            right_border = pts_white[np.array(pts_white)[:,0].argmax()]
            displacement = pt_robot - right_border
            displacement = displacement + RIGHT_BORDER_MODIFIER
            sign_x = displacement[0] / np.abs(displacement[0])
            x_error = sign_x * np.interp(np.abs(displacement[0]),
                                            [min_reading_error, max_reading_error],
                                            [0, MAX_ERROR])

        if(detect):
            pt = (right_border[0],right_border[1])
            cv2.putText(img_debug, 'RIGHT BORDER', (pt[0]+50, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            draw_cricle(img_debug,pt,clr=(0,255,0))
            d = RoadDebugItem(road_img=img_debug)
            self.debug_items.append(d)

        self.last_error = x_error
        print('X Error: ', x_error)
        return x_error
            


    def get_error_road(self,image):
        img_debug = image.copy()
        x_error = 0

        max_reading_error = image.shape[1] / 2
        min_reading_error = 25

        # centre points relative to robot camera (i.e. centre of image)
        pt_robot = np.array([image.shape[1],image.shape[0]])/2


        Y,X = np.where(np.all(image==self.road_clr,axis=2))
        detect_flag = False

        if X.size > 0 and Y.size > 0 :
            detect_flag = True
            pt_path = np.array([int(np.average(X)),int(np.average(Y))])

            displacement = pt_robot - pt_path

            sign_x = displacement[0] / np.abs(displacement[0])

            x_error = sign_x * np.interp(np.abs(displacement[0]),
                                            [min_reading_error, max_reading_error],
                                            [0, MAX_ERROR])

        if(detect_flag):
            pt = (pt_path[0],pt_path[1])
            cv2.putText(img_debug, 'ROAD DETECT', (pt[0]+50, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            draw_cricle(img_debug,pt,clr=(0,255,0))
            d = RoadDebugItem(road_img=img_debug)
            self.debug_items.append(d)


        self.last_error = x_error
        print('X Error: ', x_error)
        return x_error

class RoadDebugItem:
    def __init__(self,road_img = None, stop_img = None, error_array = None, time_array = None):
        self.road_img = road_img
        self.stop_img = stop_img
        self.err_img = get_plot_img(error_array, time_array)

main()
