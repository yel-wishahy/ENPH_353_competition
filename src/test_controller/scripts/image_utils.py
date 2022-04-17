
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

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

def contour(img,inv_thresh=-1,limit=10,num_sides=-1,area_limit=1,apply_gray=False,pre_process=False):
    """
    @brief Detects largest #limit white contours

    @params:
        img (numpy array image)
    @optional params:
        inv_threshold (integer): ignore if < 0 , useful for white contours (inverse image = inv_threshold - img)

        limit (integer): maximum number of contours to sort 

        num_sides (integer): ignore if -1, only search for contours with this many sides

        area_limit (float): maximum area as a fraction of the total img a contour can have

        gray (bool) : if True, input is gray, do not need to gray, otherwise convert input to gray
        pre_process (bool) : if True preprocess image to remove noie (blur, threshold,erode,dilate)

    @return
        cnts (list of contours/numpy arrays): list of contours sorted from largest to smallest
        centroids : list of centroid numpy arrays
    """
    cnt_img = img.copy()

    if(inv_thresh > 0 ):
        cnt_img = (inv_thresh-cnt_img)
    if(apply_gray):
        cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2GRAY)
    if(pre_process):
        # Gaussian blur
        cnt_img = cv2.GaussianBlur(cnt_img,(5,5),0)
        # Color thresholding
        _,cnt_img = cv2.threshold(cnt_img,60,255,cv2.THRESH_BINARY_INV)
        # Erode and dilate to remove accidental line detections
        cnt_img = cv2.erode(cnt_img, None, iterations=2)
        cnt_img = cv2.dilate(cnt_img, None, iterations=2)

    # Find the contours of the frame
    _,contours,_ = cv2.findContours(cnt_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    centroids = []

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

            # if(debug):
            #     print(cx,cy)
            #     print('area', cv2.contourArea(c))
            #     print(img.shape[0]*img.shape[1])
            #     cv2.line(img,(cx,0),(cx,720),(255,0,0),1)
            #     cv2.line(img,(0,cy),(1280,cy),(255,0,0),1)
            #     cv2.drawContours(img, c, -1, (0,255,0), 1)

            pt = find_centroid(c)
            centroids.append(np.array(pt))
            cnts.append(c)


    return cnts,centroids

def get_gray(img, clr='w'):
    """
    gets the grayed version of img with desired colour filter
    if clr='b' it will gray based on blue filter
    otherwise white
    """

    gray = None
    if(clr is 'b'):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        ## Gen lower mask (0-5) and upper mask (175-180)
        mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

        ## Merge the mask and crop the red regions
        mask = cv2.bitwise_or(mask1, mask2 )
        cropped = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if(clr is 'w'):
        inv = (255-img)
        gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)

    # Gaussian blur

    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Color thresholding
    ret,thresh1 = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)

    # Erode and dilate to remove accidental line detections
    mask = cv2.erode(thresh1, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


def draw_cricle(img,pt,clr=(255,0,0)):
    """
    @brief draws a circle on image at specified point, MODIFIES img
    @param pt: 2d point TUPLE as circle centre
    @param BGR clr tuple (default to (255,0,0))

    @modifies img
    @return nothing
    """
    radius = 40
    thickness = -1
    lineType = cv2.LINE_8
    shift = 0
    cv2.circle(img,pt, radius, clr, thickness, lineType, shift)


def hsv_threshold(img,clr):
    """
    @brief hsv threshold an image and return a masked version of it
    @param img : opencv image numpy mat
    @param clr: single letter colour char to identifty colour to threshold (eg. 'b' for blue or 'r' for red)
    """
    result = img.copy()

    if(clr is 'r'):
        image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 200, 200])
        upper1 = np.array([10, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,200,200])
        upper2 = np.array([180,255,255])
        
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)
        
        full_mask = lower_mask + upper_mask;
        
        result = cv2.bitwise_and(result, result, mask=full_mask)
    if(clr is 'b'):
        img_hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

        #blue clr range
        lower=np.array([100,150,0],np.uint8)
        upper=np.array([140,255,255],np.uint8)

        mask = cv2.inRange(img_hsv, lower, upper)

        result = cv2.bitwise_and(img, img, mask=mask)

    
    return result



