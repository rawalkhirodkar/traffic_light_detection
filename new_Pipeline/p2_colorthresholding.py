
import cv2, numpy as np
from copy import copy


def red_thresh(img,heatmap):
    img1 = cv2.inRange(img, np.array([0, 100, 100]), np.array([10,255,255])) #lower red hue
    img2 = cv2.inRange(img, np.array([160, 100, 100]), np.array([179,255,255])) #upper red hue
    img3 = cv2.inRange(img, np.array([160, 40, 60]), np.array([180,70,80]))
    img4 = cv2.inRange(img, np.array([0, 150, 40]), np.array([20,190,75]))
    img5 = cv2.inRange(img, np.array([145, 35, 65]), np.array([170,65,90]))
    



    img = cv2.bitwise_or(img1,img3)
    img = cv2.bitwise_or(img,img2)
    img = cv2.bitwise_or(img,img4)
    img = cv2.bitwise_or(img,img5)
    

    cv2.medianBlur(img,7)

    ret,thresh = cv2.threshold(img,127,255,0)
    heatmap = np.asarray(heatmap,dtype=np.uint8) 
    thresh_and_heatmap = cv2.bitwise_and(thresh,heatmap)
    contours, hierarchy = cv2.findContours(thresh_and_heatmap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours,thresh_and_heatmap



def heuristics(contours, original_img, min_area, min_red_density):

    annotated_image=original_img.copy()
    accepted=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        red_density = (area*1.0)/(w*h)
        width_height_ratio = (w*1.0)/h
        if area>min_area and len(approx) > 3 and red_density > min_red_density and width_height_ratio < 1.5 and width_height_ratio > 0.33:
            accepted.append(cnt)
            cv2.rectangle(annotated_image, (x,y), (x+w,y+h),(0,255,0), 2) #actual candidates in green
        else:
            cv2.rectangle(annotated_image, (x,y), (x+w,y+h),(0,0,255), 2) #rejects in blue
    return accepted,annotated_image


