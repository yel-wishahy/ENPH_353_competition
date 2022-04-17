#!/usr/bin/env python

from image_utils import clean_imgs,find_centroid,contour,hsv_threshold
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import imutils
from cnn_ocr import CharacterDetector, INPUT_SIZE, abs_path, PreProcessMode, PRE_PROCESS_MODE, CHARS
import string
from std_msgs.msg import String

camera_feed_topic = "/R1/pi_camera/image_raw"
node = 'license_detector'
template_path = abs_path + '/imgs/p_template.jpg'

FREQUENCY = 1000

def main():
    rospy.init_node(node)
    print('Started node: ' + node)
    detector = LicenseDetector()
    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        detector.process_loop()
        rate.sleep()


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
    inv_dark = (255 - (87-img)) #inverse threshold to make plate area white is 87 usually (found by testing)

    #exposed to light
    inv_light = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
    _,inv_light = cv2.threshold(inv_light, 195, 255, cv2.THRESH_BINARY)#specific threshold is 195 (tested manually)

    #contour white with area limit 
    c_out_light,_ = contour(inv_light,limit=3,area_limit=0.1,num_sides=4)
    c_out_dark,_ = contour(inv_dark,limit=3,area_limit=0.1,num_sides=4,apply_gray=True,pre_process=True)

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
        ymax = np.clip(cy+y_limits[0], cy, img.shape[0]-1)
        crops.append(img[ymin:ymax,:])

    result = hsv_threshold(img,'b')

    Y,X = np.where(np.all(result!=[0,0,0],axis=2))
    if X.size != 0 and Y.size != 0:
        cy=int(np.average(Y))
        ymin = np.clip(cy-y_limits[0], 0, cy)
        ymax = np.clip(cy+y_limits[0], cy, img.shape[0]-1)
        crops.append(img[ymin:ymax,:])

    return crops

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
            foundPlate,_= compare_template(c[:,0:int(c.shape[1]/2)],template)

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
    img_hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    #blue clr range
    lower=np.array([120,150,0],np.uint8)
    upper=np.array([140,255,255],np.uint8)

    mask = cv2.inRange(img_hsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)

    X,Y = np.where(np.all(result!=[0,0,0],axis=2))
    
    return (X.size > 0 and Y.size > 0)

def pre_process(input,shape=INPUT_SIZE,mode=PRE_PROCESS_MODE):
    """
    @brief preprocess license / id images 

    @param input : image (opencv mat) OR list of images
    @param shape (defualt : INPUT_SIZE) : desired final shape for image
    @param mode (defualt: PRE_PROCESS_MODE ) : process mode

    @return preprocessed image or list of preprocessed images
    """

    def process_img(img):
        img_out = cv2.resize(img, (shape[1],shape[0]))

        if(mode == PreProcessMode.NONE):
            return img_out.reshape(shape)

        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

        if(mode == PreProcessMode.GRAY):
            return img_out.reshape(shape)
        
        img_out = cv2.adaptiveThreshold(img_out,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,101,15)#101,15 or101, 25

        if(mode == PreProcessMode.BINARY):
            return img_out.reshape(shape)

        return img_out.reshape(shape)
    
    if(type(input) == list):
        imgs = clean_imgs(input)
        out = []
        for img in imgs:
            out.append(process_img(img))
        return out
    else:
        return process_img(input)

def compare_template(img,template,match_threshold = 0.15):
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
    binary  = pre_process(img,shape=(temp.shape[0],temp.shape[1],1),mode=PreProcessMode.BINARY)

    similarity = cv2.matchTemplate(binary,temp,cv2.TM_CCOEFF_NORMED)
    result = similarity >= match_threshold

    return result, similarity

def crop_license_chars(img_in,contour_find=True):
    """
    @brief crops characters from license plate into 4 images

    @param img_in : cv image of license plate
    
    @return list of 4 images hopefully containing the characters
    """

    #crop letters based on colour thresholding
    #gauranteed to return 4  crops of same size
    #crops less likely to have full letter
    def crop_letters_threshold(img):
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
            crops.append(c)

        return crops

    #get 4 bounds of a contour
    def get_bounds(c):
        xmin = c[:,:,0].min()
        xmax = c[:,:,0].max()
        ymin = c[:,:,1].min()
        ymax = c[:,:,1].max()

        return (xmin,xmax,ymin,ymax)

    #crop letters based on image sharpening,eroding,thresholding,contour
    #not gauranteed to return 4 letter crops, crops not same size
    #crops very likley to contain full letter!
    def crop_letters_contours(img):
        to_crop = img.copy()

        k = np.array([  [0, -1, 0],
                    [-1, 10,-1],
                    [0, -1, 0]])
        filt =  cv2.filter2D(src=img, ddepth=-1, kernel=k)

        sharp =  cv2.erode(filt, None, iterations=3)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        bi = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,3)

        cnts = cv2.findContours(bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[1:5]#ignore contour 0, keep 1-5

        unordered_crops = []

        for c in cnts:
            b = get_bounds(c)
            unordered_crops.append((b[0],to_crop[b[2]:b[3],b[0]:b[1]]))

        crops = []
        for c in sorted(unordered_crops, key=lambda val: val[0]):
            crops.append(c[1])

        return crops
    
    crops = []
    if contour_find:
        #return best of 2 crops
        crops = crop_letters_contours(img_in)
        crops = clean_imgs(crops)
    if(len(crops) == 4):
        return crops
    else:
        return crop_letters_threshold(img_in)

def crop_id_chars(img):
    """
    @brief crop character chars for id plate "P #"

    @param : id plate img (opencv mat)

    @return crops of P and #, not neccessarily full contained
    """
    dx = int(img.shape[1]/2)
    crops = [img[:,0:dx],img[:,dx:]]

    return crops

class LicenseDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_feed_topic, Image, self.license_callback, queue_size=1)

        self.latest_img = Image()
        self.count = 0

        self.template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        self.save_image(self.template,'test.jpg')
        self.empty = True
        
        self.OCR = CharacterDetector()

        self.license_plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)
        
        self.bin_debug_pub = rospy.Publisher("/license_detector/bin_debug", Image, queue_size=1)
        self.platecrop_debug_pub = rospy.Publisher("/license_detector/plates_debug", Image, queue_size=1)

        
        self.plate_debug_pub = rospy.Publisher("/license_detector/plate_crop_debug", Image, queue_size=1)
        self.chars_debug_pub = rospy.Publisher("/license_detector/chars_debug", Image, queue_size=1)

        self.plate_debugF_pub = rospy.Publisher("/license_detector/plate_debug_false", Image, queue_size=100)
        self.chars_debugF_pub = rospy.Publisher("/license_detector/chars_debug_false", Image, queue_size=100)
        self.debug_items = []
    
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
            license_bin_crop = get_license_bin_crop(self.latest_img)
            self.empty = True

            if(license_bin_crop is None):
                return

            plate_contours = None
            try:
                plate_contours,_ = find_license_plate_contours(license_bin_crop)
            except:
                return

            plate_crops = get_license_plate_crop(license_bin_crop,plate_contours,y_limits=(20,20))
            plate_crops = clean_imgs(plate_crops)
            
            if(len(plate_crops) > 0 ):
                stack =  np.concatenate(plate_crops, axis=0)
                self.platecrop_debug_pub.publish(self.bridge.cv2_to_imgmsg(stack, "bgr8"))

            license_plates,location_ids = filter_crops(plate_crops,self.template)
            license_plates = clean_imgs(license_plates)
            location_ids = clean_imgs(location_ids)

            id_to_publish = self.predict_id(location_ids)
            plate_to_publish = self.predict_plate(license_plates)

            #check debug items collected throughout process
            #print msg/publish img
            #note : consider labelling imgs in future?
            while len(self.debug_items) > 0:
                d = self.debug_items.pop()
                if(d.is_valid()):
                    print(d.get_prediction(), d.is_valid())
                    self.chars_debug_pub.publish(self.bridge.cv2_to_imgmsg(d.get_crops(),"bgr8"))
                    self.plate_debug_pub.publish(self.bridge.cv2_to_imgmsg(d.get_plate(), "bgr8"))
                else:
                    self.chars_debugF_pub.publish(self.bridge.cv2_to_imgmsg(d.get_crops(),"bgr8"))
                    self.plate_debugF_pub.publish(self.bridge.cv2_to_imgmsg(d.get_plate(), "bgr8"))
            

            self.bin_debug_pub.publish(self.bridge.cv2_to_imgmsg(license_bin_crop,"bgr8"))
            self.license_plate_pub.publish(String("Team1,pass," + id_to_publish + "," + plate_to_publish))

    def predict_id(self,ids):
        """
        @brief predict location id from a list of cropped ids
        @param ids, list of images hopefully containing 'P#'

        adds results to debug item list throughoutt the process

        @return 1 character representing optimal number
        OR '' empty character meaning no good id found

        """
        predictions = []
        best_id = ""

        for id in ids:
            flag = True
            crops = crop_id_chars(id)
            crops = pre_process(crops)

            p = self.OCR.predict_image(np.array(crops)) #list of prediction tuples : (char,confidence)

            if(p[0][0] is not 'P'):
                flag = False
            if(p[1][0] in string.ascii_uppercase or p[1][0] is '0'):
                flag = False
            if(flag):
                predictions.append(p[1])
            
            d = PlateDebugItem(id,crops,p,flag)
            self.debug_items.append(d)
        
        # best_id = sorted(predictions, key=lambda val: val[1],reverse=True)[0][0]
        if(len(predictions) > 0):
            best_id = sorted(predictions, key=lambda val: val[1],reverse=True)[0][0]

        return best_id

    def predict_plate(self,plates):
        """
        @brief predict license plate string from a list of cropped ids
        @param ids, list of images hopefully containing 'LL ##'

        adds results to debug item list throughout the process

        @return string representing 4 correct characters
        OR "" empty string meaning no good id found

        """
        predictions = [[],[],[],[]]
        best_plate = ""

        for plate in plates:
            flag = True
            crops = crop_license_chars(plate)
            crops = pre_process(crops,mode=PRE_PROCESS_MODE)

            p = self.OCR.predict_image(np.array(crops)) #list of prediction tuples : (char,confidence)

            for i in range(4):
                if( i <= 1 and p[i][0] in string.digits):
                    flag = False
                if(i > 1 and p[i][0] in string.ascii_uppercase):
                    flag = False
                if(flag):
                    predictions[i].append(p[i])
            
            d = PlateDebugItem(plate,crops,p,flag)
            self.debug_items.append(d)

        for i in range(4):
            if(len(predictions[i]) > 0):
                best_plate += sorted(predictions[i], key=lambda val: val[1],reverse=True)[0][0]
            else:
                return ""

        return best_plate

    def save_image(self,img=None,filename=None,dir=abs_path+'/imgs/license_imgs/'):
        output = self.latest_img
        name = 'img_'+str(self.count)+'.jpg'
        if(img is not None):
            output = img
        if(filename is not None):
            name = filename

        cv2.imwrite(dir+name,output)

class PlateDebugItem():
    def __init__(self,plate,crops,prediction,valid):
        self.plate = plate #associated plate crop
        self.crops = crops #list of image crops
        self.prediction = CharacterDetector.prediction_to_string(prediction) #result string with chars,confidence
        self.valid = valid #is id/plate valid?

    def get_plate(self):
        return self.plate

    def get_crops(self):
        bw_stack =  np.concatenate(self.crops, axis=0)
        clr_stack = cv2.cvtColor(bw_stack, cv2.COLOR_GRAY2BGR)
        return clr_stack
    
    def get_prediction(self):
        return self.prediction

    def is_valid(self):
        return self.valid

main()
