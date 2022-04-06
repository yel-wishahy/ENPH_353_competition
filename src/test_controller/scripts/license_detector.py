#!/usr/bin/env python

import queue
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import roslib
import os
import imutils
import copy
from queue import Queue
from cnn_tester import CharacterDetector, input_size
import string


PKG = 'test_controller'
roslib.load_manifest(PKG)
dir_path = roslib.packages.get_pkg_dir(PKG)

camera_feed_topic = "/R1/pi_camera/image_raw"
node = 'license_plate_detector'

FREQUENCY = 1000

chars = string.ascii_uppercase + string.digits

def main():
    rospy.init_node(node)
    print('Started node: ' + node)
    detector = LicenseDetector()
    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        detector.process_loop()
        rate.sleep()

def find_centroid(c):
    """
    @brief find centroid of a contour

    @param c : contour as a nd.array of 2d points

    @return a 2 point tuple (x,y)
    """
    M = cv2.moments(c)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except:
        return (0,0)
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
    @brief detects parking/license bin contour in an image and returns img cropped to that contour

    @param img , cv2 image

    @return cropped cv image
    """
    def most_frequent_clr(img):
        colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
        return colors[count.argmax()]

    def clr_equal(clr1,clr2):
        b = clr1==clr2
        return b[0] and b[1] and b[2]

    clr_light = [201,201,201]
    clr_dark = [102,102,102]


    #pre-process parking plate gray when not exposed to light
    inv_dark = (87-img) #inverse threshold to make plate area white is 87 usually (found by testing)

    #exposed to light
    inv_light = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
    _,inv_light = cv2.threshold(inv_light, 195, 255, cv2.THRESH_BINARY)#specific threshold is 195 (tested manually)

    #contour white with area limit 
    _,c_out_light = contour(inv_light,limit=3,area_limit=0.1,num_sides=4,raw=True)
    _,c_out_dark = contour(inv_dark,limit=3,area_limit=0.1,num_sides=4)

    final_crop = None

    for c_out in [c_out_dark,c_out_light]:
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
        
        if(crop.shape[0] > 0 and crop.shape[1] > 0):
            clr = most_frequent_clr(crop)
            if(clr_equal(clr,clr_light) or clr_equal(clr,clr_dark)):
                final_crop = crop

    return final_crop

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
  @param template : cv image, optional, template to compare against

  @return list of cropped images that could contain license plates
  """
  crops = []
  for c in cnts:
    cy = find_centroid(c)[1]
    ymin = np.clip(cy-y_limits[0], 0, cy)
    ymax = np.clip(cy+y_limits[0], cy, img.shape[1]-1)
    crops.append(img[ymin:ymax,:])
  
  return crops

def compare_template(img,template,match_threshold = 0.2,match_P_only=False):
    """
    @brief uses cv2.matchTemplate to compare image to reference image

    @param img : cv2 image to check
    @param templaate : cv2 image template to compare against
    @param optional match_threshold, float to dictate minimum match success

    @return bool: similar , float: match similarity
    """

    # if(img is None or template is None or 
    # img.shape[0] < 1 or img.shape[1] < 1 or 
    # template.shape[0] < 1 or template.shape[1] < 1):
    #     return False,0
    temp = template.copy()
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)
    binary = cv2.resize(binary, (temp.shape[1],temp.shape[0]))
    
    if(match_P_only):
        #crop to p only , p is on left of image
        binary = binary[:,:int(binary.shape[1]/2)]
        temp = temp[:,:int(temp.shape[1]/2)]

    similarity = cv2.matchTemplate(binary,temp,cv2.TM_CCOEFF_NORMED)
    result = similarity > match_threshold

    return result, similarity

def clean_imgs(imgs):
    """
    @brief makes sure the list of cropped images has images with valid shapes (>0)
    @return new list with only valid crops
    """
    out_list = []
    for c in imgs:
        if(c.shape[0] > 0 and c.shape[1] > 0):
            out_list.append(c)
    
    return out_list

def filter_crops(imgs,template):
    """
    @brief filters out a list of images potnetially containing license plates
        (see compare_template function, template should be a binary(black,white) image of location id: 'P#')

    @param imgs, list of cv images
    @param template , template to compare against

    @return list of imgs with potential license plates , list imgs with of location ids
    """
    candidates = clean_imgs(imgs)
    license_plates = []
    location_ids = []

    foundPlate = True
    for c in candidates:
        if(not foundPlate):
            foundPlate,_= compare_template(c,template)

        if(is_license_plate(c)):
            license_plates.append(c)
        else:
            location_ids.append(c)

    if(foundPlate):
        return license_plates, location_ids
    else:
        return [],[]

def is_license_plate(img):
    """
    @brief determines if image is a license plate image
    
    Does this by HSV colour thresholding.
    
    @return bool : is license plane
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #change this accordingly

    ## Gen lower mask (0-5) and upper mask (175-180)
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2 )
    cropped = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)


    X,Y = np.where(np.all(gray!=[0,0,0],axis=2))
    
    return (X.size > 0 and Y.size > 0)

def crop_license_letters(img):
    """
    @brief crops characters from license plate into 4 (same size) images
        Does this by binary thresholding image and finding xmin and xmax of nonzero pixels
            ~60 - dark blue license
            ~100-120 - light blue license
    
    @param img : cv image of license plate
    
    @return list of 4 images hopefully containing the characters
    """
    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,bi = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    _,bi2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    if(cv2.countNonZero(bi2) > cv2.countNonZero(bi)):
        bi = bi2

    xmax = cv2.findNonZero(bi)[:,:,0].max()
    xmin = cv2.findNonZero(bi)[:,:,0].min()
    dx = int((xmax-xmin)/5) #5 is the number of crops (4 characters + middle seperations)
    crops = []
    for i in range(1,6):
        if(i==3):
            continue
        c = img[:,xmin + (i-1)*dx:xmin+i*dx]

        c = cv2.resize(c, (input_size[1],input_size[0]))
        crops.append(c)

    return crops



class LicenseDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_feed_topic, Image, self.license_callback, queue_size=1)
        print('initialized img subscriber')
        self.latest_img = Image()
        self.count = 0
        # self.template = cv2.imread(os.path.join(dir_path, "/src/data/template.jpg"))
        self.template = cv2.imread('template.jpg', cv2.IMREAD_UNCHANGED)
        self.save_image(self.template,'test.jpg')
        self.queue = Queue()
        self.empty = True
        
        self.CR = CharacterDetector()
    
    #subscriber callback that receives latest image from camera feed
    def license_callback(self, img):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            self.empty = False
        except:
            return

    def process_loop(self):
        if(self.empty):
            return
        else:
            crop = get_license_bin_crop(self.latest_img)
            self.empty = True

            if(crop is None):
                return
            cnts = None

            try:
                cnts,_ = find_license_plate_contours(crop)
            except:
                return

            crops= get_license_plate_crop(crop,cnts,y_limits=(20,20))
            license_plates,location_ids = filter_crops(crops,self.template)

            for plate in license_plates:
                print('*****************')
                print('***FOUND MATCH***')
                print('*****************')
                cv2.imshow("Debug View",plate)
                cv2.waitKey(1)

                crops = crop_license_letters(plate)

                prediction = self.CR.predict_image(np.array(crops))
                print(prediction)
                for p in prediction:
                    argmax = np.argmax(p)
                    char = chars[argmax]
                    print(argmax,char)

                # print(predict_image(image_array))
                # print(np.argmax(predict_image(image_array)))
                # self.count = self.count + 1
                # self.save_image(plate)
                # for i in range(len(crops)):
                #     name = 'plate_' + str(self.count) + '_letter_' + str(i) + '.jpg'
                #     self.save_image(crops[i],name)

    def save_image(self,img=None,filename=None):
        output = self.latest_img
        name = 'img_'+str(self.count)+'.jpg'
        if(img is not None):
            output = img
        if(filename is not None):
            name = filename

        cv2.imwrite('imgs/'+name,output)
        print('SAVED POTENTIAL LICENSE:',name)

main()

