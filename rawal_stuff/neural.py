import cv2
import numpy as np
import random
import copy
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
# Load the Dataset of Stop Lights

np.set_printoptions(threshold=np.inf)



def create_model():
    batch_size = 32
    nb_classes = 3
    nb_epoch = 5
    data_augmentation = True

    # input image dimensions
    img_rows, img_cols = 32, 32
    img_channels = 3
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


print "Loading model"

model = create_model()
model.load_weights("traffic_light_weights.h5")



sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_heatmap = convnet('vgg_19',weights_path="model/weights/vgg19_weights.h5", heatmap=True)
model_heatmap.compile(optimizer=sgd, loss='mse')

clipnum = raw_input("Enter Clip number:\n")

f=open('../dayTrain/dayClip'+str(clipnum)+'/frameAnnotationsBULB.csv','r')
inputs=f.read()
f.close();

inputs=inputs.split()
inputs=[i.split(";") for i in inputs]
for i in range(21):
	inputs.pop(0)

traffic_light_synset = "n06874185"
ids = synset_to_dfs_ids(traffic_light_synset)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output'+str(clipnum)+'.avi',fourcc, 20.0, (1280,960))

for i in inputs:
    if i[1]=="stop":
        filename="../dayTrain/dayClip"+str(clipnum)+"/frames/"+i[0][12:len(i[0])]
       
        original_img=cv2.imread(filename)
        img=copy.copy(original_img)
        
        height, width, channels = img.shape 

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
        
        im_heatmap = preprocess_image_batch([filename], color_mode="bgr")
        out_heatmap = model_heatmap.predict(im_heatmap)
        heatmap = out_heatmap[0,ids].sum(axis=0)
        
        # print("heatmap:",np.shape(heatmap)) 
        # print(heatmap)
        my_range = np.max(heatmap) - np.min(heatmap)
        heatmap = heatmap / my_range
        heatmap = heatmap * 255
        # print(heatmap)
        
        # heat_im = np.array(heatmap = heatmap * 255, dtype = np.uint8)
        # threshed = cv2.adaptiveThreshold(heat_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        
        heatmap = cv2.resize(heatmap,(width,height))
        # cv2.imshow("heatmap1",heatmap)


        heatmap[heatmap < 128] = 0    # Black
        heatmap[heatmap >= 128] = 255 # White
        # (thresh, heatmap) = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # cv2.imshow("heatmap2",heatmap)
        # heatmap = cv2.medianBlur(heatmap,3)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
        # # cv2.medianBlur(heatmap,7)
        # ret,thresh = cv2.threshold(heatmap,127,255,0)
        # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
        
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area>20:
        #         x,y,w,h = cv2.boundingRect(cnt)                    
        #         cv2.rectangle(heatmap, (x,y), (x+w,y+h),(0,0,255), 2)
        #         cv2.imshow("heatmap",heatmap)


        img = cv2.bitwise_or(img1,img3)
        img = cv2.bitwise_or(img,img2)
        img = cv2.bitwise_or(img,img4)
        img = cv2.bitwise_or(img,img5)
        

        cv2.medianBlur(img,7)

        ret,thresh = cv2.threshold(img,127,255,0)
        # print("heatmap:",np.shape(heatmap), "img:", np.shape(thresh), type(heatmap), type(thresh))
        heatmap = np.asarray(heatmap,dtype=np.uint8) 
        thresh = cv2.bitwise_and(thresh,heatmap)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>20:
                x,y,w,h = cv2.boundingRect(cnt)                    
                if area*2.5 > w*h and w < 2 * h:
                    try:
                        fraction = 0.1
                        pad=int(fraction*w*h)
                        temp=original_img[y-pad:y+h+pad,x-pad/3:x+h+pad/3]
                        xx=cv2.resize(temp,(32,32))
                        xx=np.asarray(xx)
                        xx=np.transpose(xx,(2,0,1))
                        xx=np.reshape(xx,(1,3,32,32))
                 
                        if model.predict_classes(xx,verbose=0)==[1]:
                            cv2.rectangle(original_img, (x,y), (x+w,y+h),(255,0,0), 2)
                        else:
                            a =1
                            cv2.rectangle(original_img, (x,y), (x+w,y+h),(0,0,255), 2)

                    except:
                        pass
                               
                
        # cv2.rectangle(original_img, (x,y), (x+w,y+h),(255,0,0), 2)
        cv2.resize(original_img,(0,0),fx=0.2,fy=0.2)
        cv2.imshow("Original",original_img)
        out.write(original_img)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

cv2.destroyAllWindows()






