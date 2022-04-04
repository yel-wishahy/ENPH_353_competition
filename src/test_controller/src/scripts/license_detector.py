#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

camera_feed_topic = "/R1/pi_camera/image_raw"
node = 'license_plate_detector'

FREQUENCY = 250

def main():
    rospy.init_node(node)
    print('Started node: ' + node)
    detector = LicenseDetector()
    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        rate.sleep()

def find_centroid(c):
    """
    @brief find centroid of a contour

    @param c : contour as a nd.array of 2d points

    @return a 2 point tuple (x,y)
    """
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx,cy)

def contour(img,inv_thresh=255,limit=10,num_sides=-1,area_limit=1,raw=False,debug=False):
    """
    @brief Detects largest #limit white contours

    @params:
        img (numpy array image)
    @optional params:
        inv_threshold (integer): ignore if < 0 , useful for white contours (inverse image = inv_threshold - img)

        limit (integer): maximum number of contours to sort 

        num_sides (integer): ignore if -1, only search for contours with this many sides

        area_limit (float): maximum area as a fraction of the total img a contour can have

        raw (bool) : if True, do not preprocess image (invert, blur, hsv threshold)

    @return
        img (numpy array image): original image with contour traces and centroids marked

        c_out (list of contours/numpy arrays): list of contours sorted from largest to smallest
    """
    mask = img.copy()
    if( not raw):
        inv = (inv_thresh-img)
        if(inv_thresh < 0):
            inv = img
        gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)

        # Gaussian blur

        blur = cv2.GaussianBlur(gray,(5,5),0)

        # Color thresholding
        ret,thresh1 = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)

        # Erode and dilate to remove accidental line detections
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

    # Find the contours of the frame
    _,contours,_ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_out = []

    # Find the biggest contour (if detected)
    if len(contours) > 0:
        #sort largest to smallest by area
        c_sorted = sorted(contours,key=cv2.contourArea,reverse=True)

        #go through limit largest
        for c in c_sorted[:limit]:
            #check area limit
            if cv2.contourArea(c) >= area_limit*img.shape[0]*img.shape[1]:
                continue

            if(num_sides > 0):
                #check polygon ~ square apporximation
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.025 * peri, True)
                if len(approx) != num_sides:
                    continue
                
            cx,cy = find_centroid(c)

            if(debug):
                print(cx,cy)
                print('area', cv2.contourArea(c))
                print(img.shape[0]*img.shape[1])
                cv2.line(img,(cx,0),(cx,720),(255,0,0),1)
                cv2.line(img,(0,cy),(1280,cy),(255,0,0),1)
                cv2.drawContours(img, c, -1, (0,255,0), 1)

            c_out.append(c)


    return img,c_out

def get_license_bin_crop(img):
    """
    @brief detects parking bin contour in an image and returns img cropped to that contour

    @param img , cv2 image

    @return cropped cv image , tuple of crop bounds
    """

    #make parking plate gray the darkest colour
    inv = (87-img) #inverse threshold to make plate area white is 87 usually (found by testing)

    #contour white with area limit 0.2
    _,c_out = contour(inv,limit=3,area_limit=0.1)#num_sides=4

    crop = img.copy()
    xmin,xmax,ymin,ymax = 0,0,0,0

    #find bounds of largest contour
    #plus add any additional cropping override
    if(len(c_out) > 0):
        xmin = c_out[0][:,:,0].min()
        xmax = c_out[0][:,:,0].max()
        ymin = c_out[0][:,:,1].min()
        ymax = np.clip(c_out[0][:,:,1].max() + 100,ymin,img.shape[1]-1) #+ 100 because the license areas are seperated


    #if second area found, set to max of that
    if(len(c_out) > 1):
        ymax = c_out[1][:,:,1].max()

    #crop image with bounds
    crop = crop[ymin:ymax,xmin:xmax]

    #return cropped img and crop limits
    return crop, (xmin ,xmax,ymin,ymax)

def find_license_plate_contours(image,limit=10):
    """
    @brief detect license plate contours from parking bin white square 
    by conducting a series of image processing operations first

    @brief adapted from
    https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/#download-the-code
    

    @param image : cropped cv image , expected to containing 'P#" and license plate

    @param optional, limit , integer limit to how many contours to search for

    @return list of contours, list of square contours
    """

    #get grayscale image
    gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)

    # perform a blackhat morphological operation that will allow
    # us to reveal dark regions (i.e., text) on light backgrounds
    # (i.e., the license plate itself)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
    dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    # blur the gradient representation, applying a closing
    # operation, and threshold the image using Otsu's method
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations to clean up the
    # thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # take the bitwise AND between the threshold result and the
    # light regions of the image
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    # find contours in the thresholded image and sort them by
    # their size in descending order, keeping only the largest
    # ones
    _,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:limit]

    cnts_sqr = []
    if(len(cnts) > 0):
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            if(len(approx) != 4):
                continue
            cnts_sqr.append(c)

    # return the list of contours
    return cnts,cnts_sqr

def get_license_plate_crop(img,cnts,y_limits=(10,10)):
  """
  @brief given a set of contours likely to be contain license plate segments, crops above and below contour centroids

  @param img: opencv image with license plate
  @param cnts: list of conoute (ndarray of 2d pts)
  @param y_limits, optional: tuple of limits to crop (above,below) centroid

  @return list of cropped images that could contain license plates
  """
  crops = []
  for c in cnts:
    cy = find_centroid(c)[1]
    ymin = np.clip(cy-y_limits[0], 0, cy)
    ymax = np.clip(cy+y_limits[0], cy, img.shape[1]-1)
    crops.append(img[ymin:ymax,:])
  
  return crops


class LicenseDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_feed_topic, Image, self.license_callback, queue_size=1)
        print('initialized img subscriber')
        self.latest_img = Image()
        self.empty = True
        self.count = 0
    
    #subscriber callback that receives latest image from camera feed
    def license_callback(self, img):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            self.empty = False

            crop,_ = get_license_bin_crop(self.latest_img)
            cnts,_ = find_license_plate_contours(crop)
            crops = get_license_plate_crop(crop,cnts,y_limits=(20,20))

            for c in crops:
                try:
                    cv2.imshow("Debug View", c)
                    cv2.waitKey(1)
                except:
                    print('could not imshow')

                self.count = self.count + 1
                self.save_image(c)

        except CvBridgeError as e:
            print(e)

    def save_image(self,img=None):
        output = self.latest_img
        if(img is not None):
            output = img

        cv2.imwrite('imgs/img_'+str(self.count)+'.jpg',output)
        print('SAVED POTENTIAL LICENSE:','license_crop_',self.count)

main()

