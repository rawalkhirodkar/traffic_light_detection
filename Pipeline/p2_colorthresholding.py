
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
    temp_thresh = thresh
    # print("heatmap:",np.shape(heatmap), "img:", np.shape(thresh), type(heatmap), type(thresh))
    heatmap = np.asarray(heatmap,dtype=np.uint8) 
    thresh = cv2.bitwise_and(thresh,heatmap)
    temp_thresh2 = cv2.bitwise_and(temp_thresh,heatmap)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours,temp_thresh2



def heuristics(contours, original_img):

    annotated_image=original_img.copy()

    rejected=[]
    accepted=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        fraction = 0.1
        pad=int(fraction*w*h)
        temp=original_img[y:y+2*h,x-pad/3:x+h+pad/3]
        if area>20 and len(approx) > 3 and area*2.5 > w*h and w < 2 * h and h < 5 * w:
            accepted.append(temp)
            cv2.rectangle(annotated_image, (x,y), (x+w,y+h),(0,255,0), 2)
        else:
            rejected.append(temp)
            cv2.rectangle(annotated_image, (x,y), (x+w,y+h),(0,0,255), 2)
    return accepted,annotated_image


