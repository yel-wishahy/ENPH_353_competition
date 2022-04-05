#!/usr/bin/env python

from queue import Queue
from geometry_msgs.msg import Twist
import rospy
import cv2
from std_msgs.msg import String,Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import matplotlib.animation as animation
from multiprocessing import Pool
from multiprocessing import Process
from enum import Enum
import time

#controller node name
controller_node = "test_controller"
#competition ros topics
camera_feed_topic = "/R1/pi_camera/image_raw"
cmd_vel_topic = "/R1/cmd_vel"
license_plate_topic = "/license_plate"

#internal debug publisher topics
debug_img_topic = "/test_controller/image"
debug_err_topic = "/test_controller/error"

#debug /tune mode
DEBUG = True

#pid driving controller parameters

DRIVE_SPEED = 0.15

K_P = 2.5
K_D = 1.75
K_I = 3

FREQUENCY = 250 # #hz
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

    while not rospy.is_shutdown():
        controller.control_loop()
        rate.sleep()

    controller.stop_cmd()



class image_processor:
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
        

    #gets the grayed version of img with desired colour filter
    #if clr='b' it will gray based on blue filter
    #otherwise white
    def get_gray(self,img, clr='w'):
        if(clr is 'b'):
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            ## Gen lower mask (0-5) and upper mask (175-180) of RED
            mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
            mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

            ## Merge the mask and crop the red regions
            mask = cv2.bitwise_or(mask1, mask2 )
            cropped = cv2.bitwise_and(img, img, mask=mask)
            gray = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            return gray
        if(clr is 'w'):
            inv = (255-img)
            gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
            return gray

    #given a source image and a grayed version of it that image highlight the desired colour
    #finds the contours and returns them
    def get_contours(self,image,gray):
        image_contour = np.copy(image)
        # Gaussian blur image for noise
        blur = cv2.GaussianBlur(gray,(5,5),0)

        # Color thresholding
        ret,thresh1 = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)

        # Erode and dilate to remove accidental line detections
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find the contours of the frame
        _,contours,_ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_SIMPLE)

        return sorted(contours,key=cv2.contourArea,reverse=True)

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
        self.parking_clr = [100,0,0]#[200,100,100]#[121,19,20]
        
        self.drive_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self.license_plate_pub = rospy.Publisher(license_plate_topic, String, queue_size=1)

        self.debug_error_pub = rospy.Publisher(debug_err_topic, Float32MultiArray, queue_size=10)
        self.debug_img_pub = rospy.Publisher(debug_img_topic, Image, queue_size=10)

        self.img_processor = image_processor()

        self.last_error = 0
        self.last_time = rospy.get_time()

        self.error_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5
        self.time_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5

        self.index = 0

        self.move_state = 0
        self.last_stop_time = 0
        self.last_stop_detect = False

        #Setting an arbitrary state to start timer at
        self.state = 1
        self.drive_state = DriveState.DRIVE_FORWARD
        time.sleep(1)


    #main control loop for pid controller
    def control_loop(self):

        # #Printing state of machine to get a live display, publishing msg to start 
        # if(DEBUG):
        #     print("****State: {}".format(self.state))
        # if (self.state == 1):
        #     print("==== Start timer")
        #     self.license_plate_pub.publish(String("Team1,pass,0,AAAA"))

        # #After about 25 secs, state will reach 500. Enough time for the robot to have 
        # #moved 1m, will then stop timer; feel free to adjust
        # if self.state == 500:
        #     self.license_plate_pub.publish(String("Team1,pass,-1,AAAA"))
        #     print("==== End timer")
        #     self.state +=1
        # elif self.state < 500:
        #     self.state +=1



        if(self.img_processor.empty is False):
            image = self.img_processor.latest_img
            move = Twist()
            image_debug = image
            detected_stop = False
            detected_parking = False

            #process images for robot state
            error,image_debug = self.get_error_path(image)
            # detected_stop, image_debug = self.detect_stop(image_debug)
            # detected_parking, image_debug = self.detect_parking_bin(image_debug)

            if(detected_stop or detected_parking):
                if(self.drive_state == DriveState.DRIVE_FORWARD):
                    self.drive_state = DriveState.STOPPING
                elif(self.drive_state == DriveState.CROSSING and not self.last_stop_detect):
                    self.drive_state = DriveState.FINISHED_CROSSING
            else:
                if(self.drive_state == DriveState.FINISHED_CROSSING):
                    self.drive_state = DriveState.DRIVE_FORWARD
        

            if(self.drive_state == DriveState.DRIVE_FORWARD):
                g = self.calculate_pid(error)
                move = self.get_move_cmd(g)

            if(self.drive_state == DriveState.STOPPING):
                self.last_stop_time = rospy.get_time()
                move.linear.x = 0
                move.angular.z = 0
                self.drive_state = DriveState.STOPPED

            if(self.drive_state == DriveState.STOPPED):
                if(rospy.get_time() >= self.last_stop_time + STOP_DURATION):
                    self.drive_state = DriveState.CROSSING
                else:
                    move.linear.x = 0
                    move.angular.z = 0

            if(self.drive_state == DriveState.CROSSING or self.drive_state == DriveState.FINISHED_CROSSING):
                g = self.calculate_pid(error)
                move = self.get_move_cmd(g,1.5*DRIVE_SPEED)


            self.img_processor.empty = True
            self.last_stop_detect = detected_stop

            print("STATE: " ,self.drive_state)

            #publish error and image to debug topics
            err_pt = Float32MultiArray()
            err_pt.data = [float(error),rospy.get_time()]
            debug_img = Image()
            debug_img = self.img_processor.bridge.cv2_to_imgmsg(image_debug)
            self.debug_error_pub.publish(err_pt)
            self.debug_img_pub.publish(debug_img)

            #publish move command to robot
            self.drive_pub.publish(move)


    #get move command based on p+i+d= g
    #g only impacts angular.z
    #for now linear.x is constant
    def get_move_cmd(self,g,x_speed=DRIVE_SPEED):
        move = Twist()

        move.angular.z = g * G_MULTIPLIER
        move.linear.x =  x_speed #np.interp(np.abs(move.angular.z), [1,0], LINEAR_SPEED_RANGE)

        print('g', g)
        print('z', move.angular.z)
        print('x', move.linear.x)

        return move

    #stop command
    def stop_cmd(self):
        move = Twist()
        move.angular.z = 0
        move.linear.x = 0
        self.drive_pub.publish(move)


    #calculate pid : proportionality, integration and derivative values
    #tries to do a better job by using numpy trapezoidal integration as well as numpy gradient
    #uses error arrays for integration
    #error arrays are of size error_history 9global variable)
    def calculate_pid(self, error):
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

        print("pid ", p, i, d)
        g = p + i + d
        return g
    
    def detect_stop(self,image):
        detected = False
        Y,X = np.where(np.all(image==self.stop_clr,axis=2))

        if X.size > 0 and Y.size > 0:
            cx= int(np.average(X))
            cy= int(np.average(Y))

            if(DEBUG):
                print('detect avg red')
                cv2.line(image,(cx,0),(cx,720),(0,0,255),1)
                cv2.line(image,(0,cy),(1280,cy),(0,0,255),1)

            if(cy >= STOP_PROXIMITY_THRESHOLD):
                detected = True
                print('*****************************************')
                print('****************At Stop************')
                print('*****************************************')
                self.img_processor.save_image()
        return detected,image
    
    def detect_parking_bin(self,image):
        detected = False
        Y,X = np.where(np.all(image==self.parking_clr,axis=2))

        if X.size > 0 and Y.size > 0:
            cx= int(np.average(X))
            cy=int(np.average(Y))

            if(DEBUG):
                print('detect avg blue')
                cv2.line(image,(cx,0),(cx,720),(0,255,0),1)
                cv2.line(image,(0,cy),(1280,cy),(0,255,0),1)

            if(cx <= image.shape[1]/2 and cy >= image.shape[0]/2):
                detected = True
                print('*****************************************')
                print('****************At Bin************')
                print('*****************************************')
                self.img_processor.save_image()
        return detected, image


    
    def get_error_path(self,image):
        image_debug = image
        if(DEBUG):
            image_debug = np.copy(image)

        x_error = 0
        max_reading_error = image.shape[1] / 2
        min_reading_error = 25

        # centre points relative to robot camera (i.e. centre of image)
        pt_robot = np.array([image.shape[1],image.shape[0]])/2


        Y,X = np.where(np.all(image==self.road_clr,axis=2))

        if X.size > 0 and Y.size > 0 :
            pt_path = np.array([int(np.average(X)),int(np.average(Y))])

            displacement = pt_robot - pt_path

            sign_x = displacement[0] / np.abs(displacement[0])

            x_error = sign_x * np.interp(np.abs(displacement[0]),
                                            [min_reading_error, max_reading_error],
                                            [0, MAX_ERROR])

            if(DEBUG):
                cv2.line(image_debug,(pt_path[0],0),(pt_path[0],720),(255,0,0),1)
                cv2.line(image_debug,(0,pt_path[1]),(1280,pt_path[1]),(255,0,0),1)


        self.last_error = x_error
        print('X Error: ', x_error)
        return x_error, image_debug

    #gets largest contour points as 2d points on img
    #contours must be sorted from largest to smallest before being passed to this function
    #points are sorted based on contour area size from smallest to largest
    def get_contour_points(self,contours,clr='w',limit=10):
        #arrays for path contour centre points
        points = []
        # Find the 2 biggest contour (if detected), these should be the 2 white lines (due to inversion)
        for c in contours[:limit]:
            if clr=='b' and cv2.contourArea(c) >= 0.5*self.img_processor.img_area:
                continue
            
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            points.append((cx,cy))

        print('points', points)
        pts = np.array(points)
        print('np array points',pts)
        return pts

    def pointContourTest(self,contour,points):
        result = []

        for pt in points:
            print(pt)
            result.append(cv2.pointPolygonTest(contour,tuple(pt),measureDist=False))
        
        return np.array(result)

    #obtains error from camera feed frame by using cv2.findContours
    #
    #image is an np array, some useful techniques are applied before contour detection 
    #to make it less prone to error
    #
    #contour basically outlines the pathes/borders on either side of the road
    #from there the centre of the road to obtained relative to the 2 paths
    #
    #The error is the displacement of the robot/camera from the actual path centre
    #this displacement is mapped to a smaller value with a maximum of max_error using np.interpolate
    #the mapped displacement is returned as the error to be used for pid calculations
    def get_error_border_contour(self, image):
        xerror = 0
        image_contour = image
        if(DEBUG):
            image_contour = np.copy(image)

        max_reading_error = image.shape[1] / 2
        min_reading_error = 25

        # centre points relative to robot camera (i.e. centre of image)
        pt_robot = np.array([image.shape[1],image.shape[0]])/2

        #contour detection with white filter
        gray_w = self.img_processor.get_gray(image,clr='w')
        contours_w = self.img_processor.get_contours(image,gray_w)
        pts_w= self.get_contour_points(contours_w)

        #contour detection with blue filter
        gray_b = self.img_processor.get_gray(image,clr='b')
        contours_b = self.img_processor.get_contours(image,gray_b)
        pts_b= self.get_contour_points(contours_b,clr='b')
        pt_parking = np.array([0,0])
        if(pts_b.size > 0):
            pt_parking = pts_b[0]
            # print('*****************************************')
            # print('****************NEAR PARKING************')
            # print('*****************************************')
        
        pt_path = np.array([0,0])
        #check that more than one contour is detected
        if(pts_w.shape[0] > 1):
            # if(pts_b.shape[0] > 0):
            #     results = self.pointContourTest(contours_w[0], pts_b[:0])
            #     for r in results:
            #         if(r >= 0):
            #             xerror = 0
            #             print('****************LICENSE PLATE IGNORE************')
            #             print('****************LICENSE PLATE IGNORE************')
            #             print('****************LICENSE PLATE IGNORE************')
            # else:    
                # path centre relative to white lanes/ borders
            if(0.6*cv2.contourArea(contours_w[0]) >= cv2.contourArea(contours_w[1])):
                print('*****************************************')
                print('****************TURNING************')
                print('*****************************************')
                if(pts_w[0][0] < pts_w[1][0]):
                    xerror = -1*MAX_ERROR
                else:
                    xerror = MAX_ERROR
            else:
                pt_path = pts_w[0] + (pts_w[1] - pts_w[0])/2

                displacement = pt_robot - pt_path

                sign_x = displacement[0] / np.abs(displacement[0])

                xerror = sign_x * np.interp(np.abs(displacement[0]),
                                            [min_reading_error, max_reading_error],
                                            [0, MAX_ERROR])
            

        if(DEBUG):
            for i in range(len(pts_w[:2])):
                #for debugging only
                cv2.line(image_contour,(pts_w[i][0],0),(pts_w[i][0],720),(255,0,0),1)
                cv2.line(image_contour,(0,pts_w[i][1]),(1280,pts_w[i][1]),(255,0,0),1)

            cv2.line(image_contour,(int(pt_path[0]),0),(int(pt_path[0]),720),(0,255,0),1)
            cv2.line(image_contour,(0,int(pt_path[1])),(1280,int(pt_path[1])),(0,255,0),1)
            cv2.line(image_contour,(int(pt_parking[0]),0),(int(pt_parking[0]),720),(0,0,255),1)
            cv2.line(image_contour,(0,int(pt_parking[1])),(1280,int(pt_parking[1])),(0,0,255),1)

        self.last_error = xerror
        print('X Error: ', xerror)

        return xerror,image_contour


main()
