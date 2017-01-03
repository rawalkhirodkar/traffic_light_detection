import cv2
import numpy as np
import random
import copy
import os
# Load the Dataset of Stop Lights

videonum = raw_input("Enter Video number:\n")


for i in sorted(os.listdir("CS 490/frames/myvid"+str(videonum))):
    
    filename = "CS 490/frames/myvid"+str(videonum)+"/"+i
    print(filename)
    original_img=cv2.imread(filename)
    img=copy.copy(original_img)

    img = cv2.medianBlur(img,3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8)) # Adaptive histogram equilization
    clahe = clahe.apply(r)
    img = cv2.merge((b,g,clahe))
    
    # Threshold the HSV image to get only red colors
    img1 = cv2.inRange(img, np.array([0, 100, 100]), np.array([10,255,255])) #lower red hue
    img2 = cv2.inRange(img, np.array([160, 100, 100]), np.array([179,255,255])) #upper red hue
    img3 = cv2.inRange(img, np.array([160, 40, 60]), np.array([180,70,80]))
    img4 = cv2.inRange(img, np.array([0, 150, 40]), np.array([20,190,75]))
    img5 = cv2.inRange(img, np.array([145, 35, 65]), np.array([170,65,90]))
    

    img = cv2.bitwise_or(img1,img3)
    img = cv2.bitwise_or(img,img2)
    img = cv2.bitwise_or(img,img4)
    img = cv2.bitwise_or(img,img5)
    #order is VSH
    
    
    # img = cv2.inRange(img, np.array([min_value,min_sat,min_hue]), np.array([max_value,max_sat,max_hue]))

    cv2.medianBlur(img,7)
    
    ret,thresh = cv2.threshold(img,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    factor = 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>20:
            x,y,w,h = cv2.boundingRect(cnt)
            print(area, w*h)
            if( area * factor > w*h):
                cv2.rectangle(original_img, (x,y), (x+w,y+h),(255,0,0), 2)
            

    cv2.imshow("Before threshold",thresh)
    cv2.resize(original_img,(0,0),fx=0.2,fy=0.2)
    cv2.resize(img,(0,0),fx=0.2,fy=0.2)
    cv2.imshow("Original",original_img)
    cv2.imshow("Red Threholded", img)
    ch = 0xFF & cv2.waitKey(1000)
    if ch == 27:
        break

cv2.destroyAllWindows()
