import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os


CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 50

def detect_sky(file):

        '''Returns the image with non-sky areas masked:
        - Firstly, it classifies the pixels into sky or not sky class according to the HSV 
        value, blue and white pixels are considered sky. The problem with this approach
        is that the edges of the sky area are bad defined.
        To improve the result the edges of the image are detected and the contours are 
        found, for each area in the image the percentage of sky (given by color) is calculated,
        if more than 50% then the whole area is classified as sky
         '''
        src=cv2.imread(file)
        
        #Open as HSV...
        hsv_img=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        #Equalize value
        planes=cv2.split(hsv_img)
        planes[2]=cv2.equalizeHist(planes[2])
        mer_img=cv2.merge(planes)

        #Take blue-ish part
        lower_red = np.array([80,30,40])
        upper_red = np.array([124,255,255])
        range_img=cv2.inRange(mer_img,lower_red,upper_red)
        # plt.imshow(range_img)
        # plt.show()
        
        #White areas are also likely to be sky...
        lower_white = np.array([0,0,200])
        upper_white = np.array([255,255,255])
        range_img_2=cv2.inRange(mer_img,lower_white,upper_white)
        # plt.imshow(range_img_2)
        # plt.show()
        
        #Merging blue and white areas...
        range_img= (range_img + range_img_2)
        
        #Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        Erode_img=cv2.erode(range_img,kernel)
        Dilation_img = cv2.dilate(Erode_img, kernel)
        
        #Create mask
        ret, mask = cv2.threshold(Dilation_img, 150, 255, cv2.THRESH_BINARY)        
        notmask = cv2.bitwise_not(mask)
        masked_first = cv2.bitwise_and(src, src, mask=notmask)
        plt.imshow(masked_first)
       

        
        #Detect edges...
        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        edges = cv2.dilate(edges, None)
        #Ignore almost black areas, it is specially useful for pano images
        edges[gray<20] = 255
        line_width=int(len(src)/100)
        margin=int(len(src)/200)
        #Add rectange to image limits as edge to assure contours work
        cv2.rectangle(edges, (margin, len(src)-margin), (len(src[0])-margin, margin), 255, line_width)
        #Add image limits as edge to assure contours work
        # plt.imshow(edges)
        # plt.show()
        
        #Find contours in edges, sort by area...
        contour_info = []
        contours, hierarchy = cv2.findContours(255- edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        #Empty mask...
        mask_fin = np.zeros(gray.shape, np.uint8)
        contours_sorted = sorted(contours, key=cv2.contourArea,  reverse = True)
        
        #Evaluate each area, as they as sorted stop when too small...
        for c in contours_sorted:
            mask_c = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)
            #Value is zero for 255, 0 for no sky, get mean for countour
            mean = cv2.mean(notmask, mask=mask_c)[0]
            image_area = len(src)*len(src[0])
            per_area = cv2.contourArea(c)/image_area
            #If too small break as 
            if per_area<0.00005:
                break
            
            if mean<100:
                #If mostly sky white...
                cv2.drawContours(mask_fin, [c], -1, 255, -1)
                print(mean)
                # plt.imshow(mask_c)
                # plt.show()
             
        #Add mask to original image
        masked_img = cv2.bitwise_and(src, src, mask=mask_fin)
        
        #Edges of sky also detected
        edges_sky = cv2.Canny(mask_fin, CANNY_THRESH_1, CANNY_THRESH_2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_sky_dil = cv2.dilate(edges_sky, kernel)

        return masked_img, edges_sky_dil
        

        
