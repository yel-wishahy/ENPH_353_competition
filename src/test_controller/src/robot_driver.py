#!/usr/bin/env python
from queue import Queue

from geometry_msgs.msg import Twist
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import matplotlib.animation as animation
from multiprocessing import Pool
from multiprocessing import Process

#controller node name
controller_node = "test_controller"
#competition ros topics
camera_feed_topic = "/R1/pi_camera/image_raw"
cmd_vel_topic = "/R1/cmd_vel"

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
    def get_gray(self,img, clr):
        if(clr is 'b'):
            threshold = 20
            _, img_blue = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            inv = 255-img_blue
            gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
            return gray
        
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

        return contours



class PID_controller():
    def __init__(self):
        self.road_clr = [128, 128, 128]
        self.border_clr = [0,0,0]
        
        self.drive_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self.img_processor = image_processor()

        self.last_error = 0
        self.last_time = rospy.get_time()

        self.error_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5
        self.time_array = np.ones(ERROR_HISTORY_LENGTH) * 1e-5

        self.index = 0

        self.move_state = 0

    #main control loop for pid controller
    def control_loop(self):
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
    #points are sorted based on contour area size from smallest to largest
    #optionally draws 2 largest contour point as 2 intersecting lines on a copy of image.
    def get_contour_points(self,contours):
        #arrays for path contour centre points
        xs = []
        ys = []
        # Find the 2 biggest contour (if detected), these should be the 2 white lines (due to inversion)
        if len(contours) > 0:
            c = sorted(contours,key=cv2.contourArea)
            for i in range(1,3):
                M = cv2.moments(c[len(c)-i])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                xs.append(cx)
                ys.append(cy)

        if(len(xs) > 0 and len(ys) > 0):
            xs = np.sort(xs)
            ys= np.sort(ys)

        return xs,ys



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
        x_robot = image.shape[1] / 2
        y_robot = image.shape[0] / 2 

        gray = self.img_processor.get_gray(image,clr='w')
        contours = self.img_processor.get_contours(image,gray)
        xs,ys= self.get_contour_points(contours)
        

        x_path = 0
        y_path = 0
        #check that more than one contour is detected
        if(len(xs) > 1 and len(ys)>1):
            #path centre relative to white lanes/ borders
            x_path = xs[0] + (xs[1] - xs[0])/2
            y_path = ys[0] + (ys[1] - ys[0])/2 

            displacement_x = x_robot - x_path
            displacement_y = y_robot - y_path


            sign_x = displacement_x / np.abs(displacement_x)

            xerror = sign_x * np.interp(np.abs(displacement_x),
                                        [min_reading_error, max_reading_error],
                                        [0, MAX_ERROR])


        if(DEBUG):
            for i in range(len(xs[:2])):
                #for debugging only
                cv2.line(image_contour,(xs[i],0),(xs[i],720),(255,0,0),1)
                cv2.line(image_contour,(0,ys[i]),(1280,ys[i]),(255,0,0),1)
                cv2.drawContours(image_contour, contours, -1, (0,255,0), 1)

            cv2.line(image_contour,(int(x_path),0),(int(x_path),720),(0,255,0),1)
            cv2.line(image_contour,(0,int(y_path)),(1280,int(y_path)),(0,255,0),1)

        self.last_error = xerror
        print('X Error: ', xerror)

        return xerror,image_contour


main()
