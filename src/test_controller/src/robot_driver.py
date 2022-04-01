#!/usr/bin/env python
from queue import Queue

from geometry_msgs.msg import Twist
import rospy
import cv2
import imutils
import pytesseract
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import matplotlib.animation as animation
from multiprocessing import Pool
from multiprocessing import Process

import time

#controller node name
controller_node = "test_controller"
#competition ros topics
camera_feed_topic = "/R1/pi_camera/image_raw"
cmd_vel_topic = "/R1/cmd_vel"
license_plate_topic = "/license_plate"

#debug /tune mode
DEBUG = True

#pid parameters
K_P = 2.5
K_D = 2.5
K_I = 1

PID_FREQUENCY = 250 #hz
MAX_ERROR = 20
G_MULTIPLIER = 0.05
ERROR_HISTORY_LENGTH = 5 #array size

#vehicle speed limits for move commands
LINEAR_SPEED_RANGE = [0.1,0.25]


#main method that is executed when we rosrun or roslaunch 
def main():
    rospy.init_node(controller_node)
    controller = PID_controller()
    rate = rospy.Rate(PID_FREQUENCY)

    while not rospy.is_shutdown():
        controller.control_loop()
        rate.sleep()



class image_processor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_feed_topic, Image, self.callback, queue_size=1)
        self.latest_img = Image()
        self.empty = True
        self.frameCount = 0

    #subscriber callback that receives latest image from camera feed
    def callback(self, img):
        try:
            self.frameCount = self.frameCount + 1
            self.latest_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            self.empty = False
        except CvBridgeError as e:
            print(e)

    #gets the grayed version of img with desired colour filter
    #if clr='b' it will gray based on blue filter
    #otherwise white
    def get_gray(self,img, clr='w'):
        if(clr is 'b'):
            threshold = 20
            _, img_blue = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            inv = 255-img_blue
            gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
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
        _,contours,_ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

        return sorted(contours,key=cv2.contourArea,reverse=True)

    def get_plates(self,image,gray):
        edged = cv2.Canny(gray, 30, 200) 
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None

        for c in contours:
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
         
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
            print ("No contour detected")
        else:
             detected = 1

        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        return Cropped




class PID_controller():
    def __init__(self):
        self.road_clr = [128, 128, 128]
        self.border_clr = [0,0,0]
        
        self.drive_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self.license_plate_pub = rospy.Publisher(license_plate_topic, String, queue_size=1)
        self.img_processor = image_processor()

        self.last_error = 0
        self.last_time = rospy.get_time()

        self.error_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5
        self.time_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5

        self.index = 0

        self.move_state = 0

        #Setting an arbitrary state to start timer at
        self.state = 1
        time.sleep(1)


    #main control loop for pid controller
    def control_loop(self):

        #Printing state of machine to get a live display, publishing msg to start timer
        print("****State: {}".format(self.state))
        if (self.state == 1):
            print("==== Start timer")
            self.license_plate_pub.publish(String("Team1,pass,0,AAAA"))

        #After about 25 secs, state will reach 500. Enough time for the robot to have 
        #moved 1m, will then stop timer; feel free to adjust
        self.state +=1
        if self.state == 500:
            self.license_plate_pub.publish(String("Team1,pass,-1,AAAA"))


        if(self.img_processor.empty is False):
            image = self.img_processor.latest_img

            # error = self.get_error_off_path(image)
            # error = self.get_error_border(image)
            error,image_contour = self.get_error_border_contour(image)

            g = self.calculate_pid(error)
            move = self.get_move_cmd(g)
            self.drive_pub.publish(move)
            if(DEBUG):
                self.plot_error()
                #show image with path estimation
                cv2.imshow("Image window", image_contour)
                cv2.waitKey(1)
            self.img_processor.empty = True


    #get move command based on p+i+d= g
    #g only impacts angular.z
    #for now linear.x is constant
    def get_move_cmd(self,g):
        move = Twist()

        move.angular.z = g * G_MULTIPLIER
        move.linear.x = 0.2 #np.interp(np.abs(move.angular.z), [1,0], LINEAR_SPEED_RANGE)

        print('g', g)
        print('z', move.angular.z)
        print('x', move.linear.x)

        return move

    #plots xerror array vs. time array, to help with tuning
    def plot_error(self):
        plt.clf()
        plt.ylim((-1*MAX_ERROR,MAX_ERROR))
        plt.plot(self.time_array[:-1],self.error_array[:-1])
        plt.pause(0.000000001)
        
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

    #doesnt really work because this is from lab3 where the path environment is much simpler
    def get_error_off_path(self, image):
        xerror = 0

        # if(self.camera.frameCount <= 200):
        #     print('found first frame')
        #     print('colour from first frame is ', image[image.shape[0]/2,image.shape[1]/2])
        #     self.road_clr = image[image.shape[0]/2,image.shape[1]/2]
        #     print('gray colour reference is now', self.road_clr)

        # image/dimension bounds
        xcenter = image.shape[1] / 2
        max_reading_error = image.shape[1] / 2
        min_reading_error = 25
        h_threshold = image.shape[0] - 200

            # convert colour
            # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # image_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # cv2.imshow("Image window", image_gray)
            # cv2.waitKey(1)

            # locate path pixels based on colour
        Y, X = np.where(np.all(image == self.road_clr, axis=2))
        if X.size != 0 and Y.size != 0:
            self.move_state = 0
            displacement = xcenter - int(np.average(X))
            sign = displacement / np.abs(displacement)

            # xerror = map(np.abs(displacement), min_reading_error, 
            # max_reading_error, 0, max)
            xerror = sign * np.interp(np.abs(displacement),
                                        [min_reading_error, max_reading_error],
                                        [0, MAX_ERROR])
        else:
            xerror = self.last_error
            if self.move_state == 1:
                self.move_state = 2
            else:
                self.move_state = 1

        self.last_error = xerror

    #also doesnt really work
    def get_error_border(self, image):
        xerror = 0

        # image/dimension bounds
        xcenter = image.shape[1] / 2
        max_reading_error = image.shape[1] / 2
        min_reading_error = 25

        #show image
        # cv2.imshow("Image window", image)
        # cv2.waitKey(1)

        # locate border pixels based on colour
        Y, X = np.where(np.all(image == self.border_clr, axis=2))
        if X.size != 0 and Y.size != 0:
            self.move_state = 0
            displacement = xcenter - int(np.average(X))
            sign = displacement / np.abs(displacement)

            # max_reading_error, 0, max)
            xerror = sign * np.interp(np.abs(displacement),
                                        [min_reading_error, max_reading_error],
                                        [0, MAX_ERROR])
        else:
            xerror = self.last_error
            if self.move_state == 1:
                self.move_state = 2
            else:
                self.move_state = 1

        self.last_error = xerror
        if(xerror < 0):
            print('border LEFT')
        else:
            print('border RIGHT')
        return xerror
    
    #gets largest contour points as 2d points on img
    #contours must be sorted from largest to smallest before being passed to this function
    #points are sorted based on contour area size from smallest to largest
    def get_contour_points(self,contours):
        #arrays for path contour centre points
        points = []
        # Find the 2 biggest contour (if detected), these should be the 2 white lines (due to inversion)
        if len(contours) > 0:
            for c in contours[:10]:
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
        pts_b= self.get_contour_points(contours_b)
        
        pt_path = np.array([0,0])
        #check that more than one contour is detected
        if(pts_w.shape[0] > 1):
            # if(len(contours_b) > 0):
            #     results = self.pointContourTest(contours_b[0], pts_w[:2])
            #     for r in results:
            #         if(r >= 0):
            #             xerror = 0
            #             print('found path point on blue contour, bad!')
            # else:    
                #path centre relative to white lanes/ borders
                pt_path = pts_w[0] + (pts_w[1] - pts_w[0])/2

                displacement = pt_robot - pt_path

                sign_x = displacement[0] / np.abs(displacement[0])

                xerror = sign_x * np.interp(np.abs(displacement[0]),
                                            [min_reading_error, max_reading_error],
                                            [0, MAX_ERROR])
        
        pt_parking = np.array([0,0])
        if(pts_b.size > 0):
            pt_parking = pts_b[0]
            

        if(DEBUG):
            for i in range(len(pts_w[:2])):
                #for debugging only
                cv2.line(image_contour,(pts_w[i][0],0),(pts_w[i][0],720),(255,0,0),1)
                cv2.line(image_contour,(0,pts_w[i][1]),(1280,pts_w[i][1]),(255,0,0),1)
                cv2.drawContours(image_contour, contours_w, -1, (0,255,0), 1)

            cv2.line(image_contour,(int(pt_path[0]),0),(int(pt_path[0]),720),(0,255,0),1)
            cv2.line(image_contour,(0,int(pt_path[1])),(1280,int(pt_path[1])),(0,255,0),1)
            cv2.line(image_contour,(int(pt_parking[0]),0),(int(pt_parking[0]),720),(0,0,255),1)
            cv2.line(image_contour,(0,int(pt_parking[1])),(1280,int(pt_parking[1])),(0,0,255),1)

        self.last_error = xerror
        print('X Error: ', xerror)

        return xerror,image_contour


main()
