# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#               The following module is used to found colombian plates of
#               particular vehicles in images (yellow plates) and perform 
#               an OCR using the wrapper package of Google Tesseract
#               for python. 
#      
#   USAGE:
#                1. All packages specified in the "requirements.txt" file must
#                be installed in advance.
#                2. "filename" variable must have the directory to the 
#                dataset to be evaluated.
#                3. Desired functions can be enables using the global variables
#                declared with mayus. By default all images are showed for 
#                every step and a key is expected to be pressed between 
#                images.
#        
#   NOTES:
#                Please consider that the module performs above 90% accurately
#                at finding the location of the plate in the image, and above
#                80% at detecting single characters when a AAA111 type of 
#                string is detected. But still needs improvement in the image
#                preprocessing steps of the detected plate before using the
#                OCR engine.
#
#   AUTHORS:
#               David Calles, Carolina Mercado, Mateo Gomez 
# 
#   ACTION:                         DATE:           NAME:
#    First implementation           1/Dic/2020      All
#    Code comments and upload       3/Dic/2020      David Calles                
#      
##############################################################################
"""

#----------------------------------------------------------------------------#
#-----------------------EXTERNAL PACKAGES DEFINITIONS------------------------#
#----------------------------------------------------------------------------#
import cv2
import numpy as np
import pytesseract
import glob
import re

#----------------------------------------------------------------------------#
#-----------------------------FUNCTIONS DEFINITIONS--------------------------#
#----------------------------------------------------------------------------#

"""***************************************************************************   
# NAME: hsv_filter
# DESCRIPTION: Manually tuned hsv filter with morphological operations
#               to segment color in image similar to the ones of a plate.
#
# PARAMETERS: ima_filt: input image to be filtered (not modified)
#                
# RETURNS:    filtered: output filtered image of same size as inpuy
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  28/Nov/2020      All
***************************************************************************"""
def hsv_filter(ima_filt):
    # DEFINE values for HSV filter
    lower = np.array([35/2,90,90])
    upper = np.array([90/2,255,255])
    hsv = cv2.cvtColor(ima_filt, cv2.COLOR_BGR2HSV)
    # THRESHOLS: HSV image to get only red colors
    hsv_mask = cv2.inRange(hsv.copy(), lower, upper)
    # APPLY: Morfologic operations
    struct_elem_mask = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    closing_mask = cv2.morphologyEx(hsv_mask.copy(), 
                                    cv2.MORPH_CLOSE,struct_elem_mask)
    opening_mask = cv2.morphologyEx(closing_mask,
                                    cv2.MORPH_OPEN,struct_elem_mask)
    filtered = cv2.bitwise_and(ima_filt,ima_filt, mask=opening_mask)
    return filtered

"""***************************************************************************   
# NAME: auto_canny
# DESCRIPTION: Canny border detection with statistically generated 
#               upper and lower thresholds.
#
# PARAMETERS: image: input image to detect borders (not modified)
#             sigma: sigma value for calculating thresholds depending on 
#                    median value.
#                
# RETURNS:    edged: output filtered image of same size as inpuy
# 
# Taken from: https://www.pyimagesearch.com/2015/04/06/zero-parameter\
#            -automatic-canny-edge-detection-with-python-and-opencv/ 
#             
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  28/Nov/2020      All
***************************************************************************"""
def auto_canny(image, sigma=0.99):
	# COMPUTE the median of the single channel pixel intensities
	v = np.median(image)
	# APPLY automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# RETURN the edged image
	return edged


"""***************************************************************************   
# NAME: order_points
# DESCRIPTION: Order a set of 4 points in clockwise order
#
# PARAMETERS: pts: input array of 4 points in image
#                
# RETURNS:    rect: ordered array of 4 points 
#             
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  28/Nov/2020      All
***************************************************************************"""
def order_points(pts):
    
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


"""***************************************************************************   
# NAME: perspective_transform
# DESCRIPTION: Perform a perspective transform of a plate
#
# PARAMETERS: img: original image where the plate is located
#             bound: 4 points of bounding box 
#                            
# RETURNS:    warped: result from perspective transformation of plate
#             plate_2: Corners used in perspective transform, 
#                     (1,2,3,4) = (red,blue,green,black)
#             
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  30/Nov/2020      All
***************************************************************************"""
def perspective_transform(img, bound):    
    # DEFINE SHAPE OF IMAGE
    h,w,c = img.shape
    # DEFINE original image coordinates
    img_corners= np.float32(np.array([[0,0],
                           [w-1,0],
                           [w-1, h-1],
                           [0, h-1]]))
    
    # ORDER points clock-wise
    bound2 = order_points(bound)
    
    # INCREASE size of detected bounding box by xt,yt pixels
    plate_2 = img.copy()
    xt = 0
    yt = 0
    bound3 = np.array([[bound2[0,0]-xt, bound2[0,1]-yt], 
                       [bound2[1,0]+xt, bound2[1,1]-yt],
                       [bound2[2,0]+xt, bound2[2,1]+yt],
                       [bound2[3,0]-xt, bound2[3,1]+yt]], dtype=np.uint16)
    
    # DRAW corners of bounding box
    cv2.circle(plate_2, tuple(bound3[0,:]), 4,(0,0,255),-1)
    cv2.circle(plate_2, tuple(bound3[1,:]), 4,(255,0,0),-1)
    cv2.circle(plate_2, tuple(bound3[2,:]), 4,(0,255,0),-1)
    cv2.circle(plate_2, tuple(bound3[3,:]), 4,(0,0,0),-1)
    
    
    bound3_f = np.float32(bound3)
    
    # GET perspective transformation matrix
    T = cv2.getPerspectiveTransform(bound3_f, img_corners) 
    
    # APPLY perspective transform 
    warped = cv2.warpPerspective(img.copy(), T, (w,h))
    
    # RETURN warped image and original image+corners of bounding box
    return warped, plate_2

#----------------------------------------------------------------------------#
#---------------------------- GLOBAL VARIABLES ------------------------------#
#----------------------------------------------------------------------------#
    
JUSTFIRST = False    # Use just first image
SHOW_ANY_IMG = True # Must be enabled to show any image
SAVE = False     # Must be enabled to save "last images" used
SAVE_PATH = "INFORME/"
# ENABLE showing images from different steps from the process
SHOW_ORIGINAL = True
SHOW_BILATERAL = True
SHOW_DENOISE = True
SHOW_THRESH = True
SHOW_CANNY = True
SHOW_ALL_CONTOURS = True
SHOW_ALL_BBOX = True
SHOW_5_BBOX = True
SHOW_RECTANGLES = True
SHOW_WARPED = True
SHOW_WARPED_POINTS = True
SHOW_WARPED_BINARIZED = True
SHOW_WARPED_DILATED = True

# DEFINE size of image to be showed. (wont affed original data)
SHOW_SIZE = 0.7
IMG_WAITKEY = 0 #Time in milisecods between images (if 0 = MANUAL)

# GET image filenames
#filename = "PARTICULAR/"
filename = "PARTICULAR2/"
#filename = "INTERNETTT/"
names = glob.glob(filename+"*.jpg")
print("Images detected: ", len(names))
iterations = range(len(names))


# USE just first image if enabled
if JUSTFIRST:
    iterations = range(1)

# DEFINE original-estimated plates
real_plates = []
obtained_plates = []  
    
#----------------------------------------------------------------------------#
#------------------------ ITERATE TRHOUGH IMAGES ----------------------------#
#----------------------------------------------------------------------------#
    
for im in iterations:
    
    # GET original plate value from filename
    plateOriginal = names[im][len(filename):len(filename)+6]
    real_plates.append(plateOriginal)
    print("Iteration: ", im)
    print("\t current plate = " +str(plateOriginal))
    
    # READ BGR image
    img = cv2.imread(names[im])
    
    # FILTER noise
    img_clean =cv2.bilateralFilter(img.copy(), -1, 5, 5)
    
    # SEGMENT potential plates by color
    th = hsv_filter(img_clean)
    
    # DETECT borders using canny and statistical thresholds
    edges = auto_canny(th)

    # FIND contours
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # DRAW contours (with random colors)
    rand_color = np.uint8(np.random.randint(255, size=(len(contours), 3)))
    all_contours = img.copy()
    for f in range(len(contours)):
        cv2.drawContours(all_contours, contours, f,
                         (int(rand_color[f,0]), int(rand_color[f,1]),
                          int(rand_color[f,2])), 2)
        
    # GET bounding boxes for every contour                                    
    rectangles_pre = []
    for j in range(len(contours)):
        rect = cv2.minAreaRect(contours[j])
        approx = cv2.boxPoints(rect)
        approx = np.int0(approx)
        rectangles_pre.append(approx.reshape(4,1,2))
        
    # DRAW bounding boxes of all contours
    area_contours = cv2.drawContours(img.copy(), rectangles_pre,
                                       -1, (0, 0, 255), 2)    
    
    # FILTER 5 contours with biggest bounding box area
    contours5 = sorted(rectangles_pre,key=cv2.contourArea, reverse = True)[:5] 
    
    # DRAW filtered bounding boxes
    area5_contours = cv2.drawContours(img.copy(), contours5,
                                       -1, (0, 255, 0), 2)
    
    
    # GET bounding box with best aspect ratio = 2
    rectangles_img = img.copy()
    rectangles = []
    error = 100000000
    for i in range(len(contours5)):
        # GET bounding box of contour (rotation allowed)
        rect = cv2.minAreaRect(contours5[i])
        approx = cv2.boxPoints(rect)
        approx = np.int0(approx)
        # ACCUMULATE results
        rectangles.append(approx.reshape(4,1,2))
        # GET bounding box of bounding box (rotation not allowed)
        [x, y, w, h] = cv2.boundingRect(approx)
        # CALCULATE aspect ratio and error to theoretical value (2)
        aspect = w/h
        aspect_error = abs(2-aspect)    
        # KEEP bounding box with least error
        if(aspect_error < error):            
            error = aspect_error
            the_rectangle = approx.reshape(4,1,2)
            
    # DRAW resulting plate
    the_rectangle2 = (the_rectangle, the_rectangle)      
    #cv2.drawContours(rectangles_img, rectangles, -1, (255, 0, 0), 2)
    cv2.drawContours(rectangles_img, the_rectangle2, -1, (255, 255, 0), 6)
    
    # PERFORM perspective transform of detected plate 
    warped, plate_image = perspective_transform(img.copy(),
                                        np.int32(the_rectangle.reshape(4,2)))
    
    # COLOR change BGR -> GRAY
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # BINARIZATION of image
    thresh = np.mean(warped_gray)*0.68
    ret,th2 = cv2.threshold(warped_gray,thresh,255,cv2.THRESH_BINARY)#100 sirve
    
    # MORPHOLOGIC 
    struct_elem_mask = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilate_mask = cv2.dilate(th2.copy(),struct_elem_mask)
    
    #------------------------ SHOW IMAGES --------------------------

    if SHOW_ANY_IMG:    
        if SHOW_ORIGINAL:
            img_show = cv2.resize(img, None, fx=SHOW_SIZE, fy=SHOW_SIZE)
            cv2.imshow("Original img", img_show)
        if SHOW_DENOISE:
            sharpen_show = cv2.resize(img_clean, None, fx=SHOW_SIZE,
                                      fy=SHOW_SIZE)
            cv2.imshow("Denoised img", sharpen_show)
        if SHOW_THRESH:
            th_show = cv2.resize(th, None, fx=SHOW_SIZE, fy=SHOW_SIZE)
            cv2.imshow("Thresh Edges img", th_show)
        if SHOW_CANNY:
            edges_show = cv2.resize(edges, None, fx=SHOW_SIZE,
                                    fy=SHOW_SIZE)
            cv2.imshow("Canny Edges img", edges_show)
        if SHOW_ALL_CONTOURS: 
            contours_show = cv2.resize(all_contours, None, fx=SHOW_SIZE,
                                       fy=SHOW_SIZE)
            cv2.imshow("Contours img", contours_show)
        if SHOW_ALL_BBOX: 
            bbox_show = cv2.resize(area_contours, None, fx=SHOW_SIZE,
                                       fy=SHOW_SIZE)
            cv2.imshow("All Bounding boxes img", bbox_show)
        if SHOW_5_BBOX: 
            contours_show10 = cv2.resize(area5_contours, None, fx=SHOW_SIZE,
                                         fy=SHOW_SIZE)
            cv2.imshow("5 Boungind boxes (biggest area)", contours_show10)
        if SHOW_RECTANGLES: 
            rectangles_show = cv2.resize(rectangles_img, None, fx=SHOW_SIZE,
                                         fy=SHOW_SIZE)
            cv2.imshow("Rectangles img", rectangles_show)
        if SHOW_WARPED: 
            warped_show = cv2.resize(warped, None, fx=SHOW_SIZE, fy=SHOW_SIZE)
            cv2.imshow("Warped img", warped_show)          
        if SHOW_WARPED_POINTS:
            plate_show = cv2.resize(plate_image, None, fx=SHOW_SIZE,
                                    fy=SHOW_SIZE)
            cv2.imshow("plate img", plate_show)
        if SHOW_WARPED_BINARIZED: 
            otsu_show = cv2.resize(th2, None, fx=SHOW_SIZE, fy=SHOW_SIZE)
            cv2.imshow("wapred binarized img", otsu_show)
        if SHOW_WARPED_DILATED: 
            dilate_show = cv2.resize(dilate_mask, None, fx=SHOW_SIZE,
                                     fy=SHOW_SIZE)
            cv2.imshow("warped dilated img", dilate_show)
        
    cv2.waitKey(IMG_WAITKEY)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

        
    # OCR: Text detection
    text = pytesseract.image_to_string(dilate_mask,config='--psm 11')
    text2 = text
    
    # ELIMINATE special chacaters ':', '-', ' ', '\n'
    text2 = text2.translate({ord(i): None for i in '_,.[]{}():/- \n'})

    # SELECT strings of 3 numbers and 3 digits (e.g ABC123)
    regex_obj = re.compile(r'(\D\D\D\d\d\d)')
    mo = None
    mo = regex_obj.search(text2)
    
    # VERIFY  if selected format (ABC123) was detected
    if mo is not None:
        print("\t detected plate= ", mo.group(0))
        obtained_plates.append(mo.group(0))
    else:
        print("\t Plate could not be detected accurately!")
        obtained_plates.append("------")

# CALCULATE error from obtained plates
total_plates = len(real_plates)
correct = 0
incorrect = 0
not_detected = 0
chars_correct = 0
chars = 0

print ("-------------------------------------------------")
print ("ORIGINAL \t \t ESTIMATED")

for i in range(total_plates):
    print("{}\t  ||\t\t{}".format(real_plates[i], obtained_plates[i]))
    if (obtained_plates[i]=="------"):
        not_detected += 1
    elif ((obtained_plates[i]==real_plates[i].upper()) or 
          (obtained_plates[i]==real_plates[i].lower())):
        correct += 1
        # GET error per character in correct images
        for j in range(6):
            chars+=1
            if ((obtained_plates[i][j]==real_plates[i][j].upper()) or 
                (obtained_plates[i][j]==real_plates[i][j].lower())):
                chars_correct += 1
    else:
        incorrect += 1
        for j in range(6):
            chars+=1
            # GET error per character in incorrect images
            if ((obtained_plates[i][j]==real_plates[i][j].upper()) or 
                (obtained_plates[i][j]==real_plates[i][j].lower())):
                chars_correct += 1
try:
    char_precision = 100*chars_correct/chars 
except:
    char_precision = 0
print ("-------------------------------------------------")
print ("TOTAL PLATES PROCESSED: ", total_plates)    
print ("TOTAL PLATES DETECTED: ", total_plates-not_detected) 
print ("CORRECT PLATES ", correct)    
print ("INCORRECT PLATES ", incorrect) 
print ("CORRECT CHARACTERS = {}/{} = {}".format(chars_correct,
                                         chars, char_precision))
                                        
# SAVE image  
if SAVE:
    cv2.imwrite(SAVE_PATH+"Original.jpg", img)    
    cv2.imwrite(SAVE_PATH+"Bilateral.jpg", img_clean) 
    cv2.imwrite(SAVE_PATH+"HSV_Filtered.jpg", th) 
    cv2.imwrite(SAVE_PATH+"Canny.jpg", edges)
    cv2.imwrite(SAVE_PATH+"All_contours.jpg", all_contours)
    cv2.imwrite(SAVE_PATH+"All_bbox.jpg", area_contours)
    cv2.imwrite(SAVE_PATH+"Area_5_bbox.jpg", area5_contours)
    cv2.imwrite(SAVE_PATH+"Rectangle.jpg", rectangles_img)
    cv2.imwrite(SAVE_PATH+"Warped.jpg", warped)
    cv2.imwrite(SAVE_PATH+"Warped_Binarized.jpg", th2)
    cv2.imwrite(SAVE_PATH+"Warped_Dilated.jpg", dilate_mask)
    
    
    
    
        