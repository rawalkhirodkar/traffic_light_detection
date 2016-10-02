import time
start=time.time()
from math import fabs 
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
print "Theano initialization: " , time.time() - start, " s"
from p0_preprocessing import *
from p1_VGG import *
from p2_colorthresholding import *
from p3_SVC import *
from p3_NN import *
from collections import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist=np.ravel(axeslist)
        axeslist[ind].imshow(figures[title], cmap=plt.gray())
        axeslist[ind].set_title(title)
        axeslist[ind].set_axis_off()
    #plt.tight_layout() # optional
    plt.show()



def ret_area(a, b):  # return area overlap of two rectangles, returns 0 if rectangles don't intersect
    if(a.xmax < b.xmin) or (b.xmax < a.xmin) or (a.ymax < b.ymin) or (b.ymax < a.ymin):
        return 0

    dx = fabs(min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    dy = fabs(min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    return dx*dy




def run_pipeline(filename,BBs,ids,model_heatmap,model,red_density):
    st=time.time()
    accepted=[]


    # Initialization

    filename="../"+filename

    ######################## Stage 0



    t=time.time()

    original_image, processed_image = preprocess(cv2.imread(filename), 7.0, (8,8))
    processed_image_BGR=cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)

    print "Stage0: Preprocessing: ",time.time()-t," s"
    t=time.time()

    ######################## Stage 1

    height, width, channels = original_image.shape

    heatmap =  generate_heatmap(filename,model_heatmap,ids,width,height,128)  #Takes original image as input,not the preprocessed one
    #Ideally we should pass preprocessed one, but will need a disk write!


    print "Stage1: VGG Heatmap: ",time.time()-t," s"
    t=time.time()

    ######################## Stage 2

    contours, thresh = red_thresh(processed_image, heatmap) # Performs red thresholding on HSV Space, draws BBs over the contours
    heuristics_accept, annotated_image = heuristics(contours, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), 20, red_density) #last two arg are minarea, red density
    print "Stage2: Color Thresholding: ",time.time()-t," s"

    t=time.time()


    ######################## Stage 3

    #accepted, final_annotated_image = SVC(accepted,annotated_image,original_image,clf) # Removes false positives using a classifier
    classifier_results,final_annotated_image = NN(heuristics_accept,annotated_image,original_image, model) # Removes false positives using a classifier
    print "Stage3: Classifier: ",time.time()-t," s"

    t=time.time()

    ######################## Finally
    total_lights=len(BBs)
    false_pos=0
    true_pos=0
    results = [0]*len(heuristics_accept) #boolean for whether classification was correct or not
    predicted_pos = sum(classifier_results) #total positives predicted

    i = 0
    for cnt in heuristics_accept:
        if classifier_results[i] == 1:   
            x,y,w,h = cv2.boundingRect(cnt)
            rguess = Rectangle(x, y, x+w, y+h)
            for bb in BBs:
                rtrue=Rectangle(bb[0],bb[1],bb[2],bb[3])
                ratio = ret_area(rguess,rtrue)/ret_area(rtrue,rtrue)
                if ratio>0.5:
                    true_pos+=1
                    results[i] = 1
                    break
        i += 1

    false_pos = predicted_pos - true_pos
    precision = 0
    recall = 0
    if(predicted_pos > 0):
    	precision = (true_pos*1.0)/predicted_pos

    if(total_lights > 0):
    	recall = (true_pos*1.0)/total_lights

    print "Frame Performance(TP,FP,LIGHTS): ",true_pos,false_pos,total_lights
    print "Frame Precision: ", precision
    print "Frame Recall", recall
    # Visualize

    D=OrderedDict({"1) Original":cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),"2) Heatmap":heatmap,"3) Color Thresholded":thresh, "4) Annotated Image":final_annotated_image})


    #plot_figures(D,2,2)

    return t - st, original_image,heatmap,thresh,final_annotated_image,total_lights,true_pos,false_pos

