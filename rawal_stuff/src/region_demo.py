import cv2
import numpy as np
import random
import copy
import dlib

from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids
np.set_printoptions(threshold=np.inf)



#----------------------------Globals------------------------------------------------------------
MIN_AREA = 20
MAX_AREA = 500
MIN_RED_DENSITY = 0.4
MIN_BLACk_DENSITY_BELOW = 0
MIN_POLYAPPROX = 3
WIDTH_HEIGHT_RATIO = [0.333, 1.5] #range

#------------------------------------------------------------------------------------------------
tracker_list = []
TRACK_FRAME = 10
VOTE_FRAME = 3

frame0_detections = []
frame1_detections = []
frame2_detections = []
frame_detections = []

RADIAL_DIST = 10
#------------------------------------------------------------------------------------------------
def dist(x1,y1,x2,y2):
    a = np.array((x1 ,y1))
    b = np.array((x2, y2))
    return np.linalg.norm(a-b)
#------------------------------------------------------------------------------------------------
BOUNDING_BOX = [0,0,0,0] #x1, y1, x2, y2
#------------------------------------------------------------------------------------------------
def prune_detection(detections):
    ans = []
    size = len(detections)
    for i in range(0,size):
        (x,y,w,h) = detections[i]
        found = -1
        for j in range(i+1,size):
            (x1,y1,w1,h1) = detections[j]
            if(dist(x,y,x1,y1) < RADIAL_DIST):
                found = 1
                break
        if found == -1:
            ans.append(detections[i])
    return ans
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def inside(p):
    (x,y) = p
    if(x < BOUNDING_BOX[2] and x > BOUNDING_BOX[0] and y < BOUNDING_BOX[3] and y > BOUNDING_BOX[1]):
        return True
    return False
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def is_violation(frame_detections):
    for (x,y,w,h) in frame_detections:
        p1 = (x,y)
        p2 = (x+w,y)
        p3 = (x,y+h)
        p4 = (x+w,y+h)
        if(inside(p1) and inside(p2) and inside(p3) and inside(p4)):
            continue
        elif(not(inside(p1)) and not(inside(p2)) and not(inside(p3)) and not(inside(p4))):
            continue
        else:
            return True
    return False
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

def create_model():
    nb_classes = 2
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 128, 128), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

#------------------------------------------------------------------------------------------------


print "Loading model"
model = create_model()
model.load_weights("../model/traffic_light_weights.h5")


#------------------------------------------------------------------------------------------------
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_heatmap = convnet('vgg_19',weights_path="../model/weights/vgg19_weights.h5", heatmap=True)
model_heatmap.compile(optimizer=sgd, loss='mse')
traffic_light_synset = "n06874185"
ids = synset_to_dfs_ids(traffic_light_synset)
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
VIDEO_NUM = raw_input("Enter Video number:\n")
VIDEO_PATH = "../ravi_data/myvideo"+str(VIDEO_NUM)+".mp4"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output'+str(VIDEO_NUM)+'.avi',fourcc, 20.0, (1280,800))
cap = cv2.VideoCapture(VIDEO_PATH)
#------------------------------------------------------------------------------------------------

frame_num = -1
VIOLATION = -1

while(cap.isOpened()):
    ret, original_img = cap.read()
    img=copy.copy(original_img)
    
    height, width, channels = img.shape
    
    if(frame_num == -1):
        center_x = width/2
        center_y = height/2 
        BB_width = width/4
        BB_height = height/4
        BOUNDING_BOX = [center_x-BB_width,center_y-BB_height,center_x + BB_width, center_y + BB_height ]

    frame_num += 1
    #------------------detection begins--------------------------------------------------------  
    if(frame_num % TRACK_FRAME < VOTE_FRAME): #VOTE_FRAME = 3, then 0,1,2 allowed
        #------------------reset------------------------    
        if(frame_num % TRACK_FRAME == 0):
            tracker_list = []
            frame0_detections = []
            frame1_detections = []
            frame2_detections = []
        #------------------reset------------------------ 

        #-----------preprocess------------------------------------
        img = cv2.medianBlur(img,3) # Median Blur to Remove Noise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        
        b,g,r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8)) # Adaptive histogram equilization
        clahe = clahe.apply(r)
        img = cv2.merge((b,g,clahe))
        #----------------------------------------------------------
        
        #----------red threshold the HSV image--------------------
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
        #----------------------------------------------------------

        #--------------------Heatmap------------------------------------
        cv2.imwrite("temp.png",original_img)
        im_heatmap = preprocess_image_batch(["temp.png"], color_mode="bgr")
        out_heatmap = model_heatmap.predict(im_heatmap)
        heatmap = out_heatmap[0,ids].sum(axis=0)
        my_range = np.max(heatmap) - np.min(heatmap)
        heatmap = heatmap / my_range
        heatmap = heatmap * 255        
        heatmap = cv2.resize(heatmap,(width,height))
        heatmap[heatmap < 128] = 0    # Black
        heatmap[heatmap >= 128] = 255 # White
        heatmap = np.asarray(heatmap,dtype=np.uint8) 

        #----------------------------------------------------------
        thresh = cv2.bitwise_and(thresh,heatmap)
        #----------------------------------------------------------
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            red_density = (area*1.0)/(w*h)
            width_height_ratio = (w*1.0)/h 
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            temp=cv2.cvtColor(original_img[y+h:y+2*h,x:x+w], cv2.COLOR_RGB2GRAY)
            (thresh, temp) = cv2.threshold(temp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            black_density_below = ((w*h - cv2.countNonZero(temp))*1.0)/(w*h)
            if area>MIN_AREA and area < MAX_AREA and len(approx) > MIN_POLYAPPROX and red_density > MIN_RED_DENSITY and width_height_ratio < WIDTH_HEIGHT_RATIO[1] and width_height_ratio > WIDTH_HEIGHT_RATIO[0] and black_density_below > MIN_BLACk_DENSITY_BELOW:
                try:
                    r_x1=x-50
                    r_y1=y-50
                    r_x2=x+w+50
                    r_y2=y+h+50
                    temp=original_img[r_y1:r_y2,r_x1:r_x2]
                    xx=cv2.resize(temp,(128,128))
                    xx=np.asarray(xx)
                    xx=np.transpose(xx,(2,0,1))
                    xx=np.reshape(xx,(1,3,128,128))
             
                    if model.predict_classes(xx,verbose=0)==[1]:
                        cv2.rectangle(original_img, (x,y), (x+w,y+h),(0,255,0), 2)
                        #append detections
                        if frame_num % TRACK_FRAME == 0:
                            frame0_detections.append((x,y,w,h))
                        elif frame_num%TRACK_FRAME == 1:
                            frame1_detections.append((x,y,w,h))
                        elif frame_num%TRACK_FRAME == 2:
                            frame2_detections.append((x,y,w,h))
                        
                    else:
                        cv2.rectangle(original_img, (x,y), (x+w,y+h),(255,0,0), 1)

                except Exception as e:
                    cv2.rectangle(original_img, (x,y), (x+w,y+h),(0,255,0), 2) #edges are allowed
                    print e
                    pass
        	# cv2.rectangle(original_img, (x,y), (x+w,y+h),(39,127,255), 3)

        #--------------------Violation in Detect Phase------------------------------
        frame_detections = []
        if(frame_num % TRACK_FRAME == 0):
            frame_detections = frame0_detections
        if(frame_num % TRACK_FRAME == 1):
            frame_detections = frame1_detections
        if(frame_num % TRACK_FRAME == 2):
            frame_detections = frame2_detections
        #--------------------Violation in Detect Phase------------------------------

        #compute and start tracking
        if frame_num % TRACK_FRAME == 2:
            all_detections = frame0_detections + frame1_detections + frame2_detections
            final_detections = prune_detection(all_detections)
            for (x,y,w,h) in final_detections:
                tracker = dlib.correlation_tracker()
                tracker.start_track(original_img, dlib.rectangle(x,y,(x+w),(y+h)))
                tracker_list.append(tracker)
    #------------------detection end----------------------------------------------------                   
   
    #------------------tracking begins----------------------------------------------------                   

    else:
        frame_detections = []
        for tracker in tracker_list:
            tracker.update(original_img)
            rect = tracker.get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(original_img, pt1, pt2, (255, 255, 255), 2)
            frame_detections.append((pt1[0], pt1[1], pt2[0]-pt1[0], pt2[1]-pt1[1]))

    #------------------ tracking end---------------------------------------------------- 
    if(is_violation(frame_detections) == True):
        cv2.rectangle(original_img, (BOUNDING_BOX[0],BOUNDING_BOX[1]), (BOUNDING_BOX[2],BOUNDING_BOX[3]),(0, 0, 255), 2)
    else:
        cv2.rectangle(original_img, (BOUNDING_BOX[0],BOUNDING_BOX[1]), (BOUNDING_BOX[2],BOUNDING_BOX[3]),(60, 255, 255), 2)
    original_img = cv2.resize(original_img,(1280,800))
    cv2.imshow("Annotated",original_img)
    out.write(original_img)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------
