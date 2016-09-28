import time
start=time.time()
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



def ret_area(a, b):  # return area overlap of two triangles, returns 0 if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0




def run_pipeline(filename,BBs,ids,model_heatmap,model):
    st=time.time()
    accepted=[]


    # Initialization

    filename="../"+filename
    
    ''' 
    import cPickle
    if model=="SVC":
        with open('my_dumped_classifier.pkl', 'rb') as fid:
            clf = cPickle.load(fid)
    '''
        

    ######################## Stage 0



    t=time.time()

    original_image, processed_image = preprocess(cv2.imread(filename), 16.0, (128,128))
    processed_image_BGR=cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)

    print "Stage0: Preprocessing: ",time.time()-t," s"
    t=time.time()

    ######################## Stage 1

    height, width, channels = original_image.shape

    heatmap =  generate_heatmap(filename,model_heatmap,ids,width,height,128)  # Takes original image as input, not the preprocessed one


    print "Stage1: VGG Heatmap: ",time.time()-t," s"
    t=time.time()

    ######################## Stage 2

    contours, thresh = red_thresh(processed_image, heatmap) # Performs red thresholding on HSV Space, draws BBs over the contours
    accepted, annotated_image = heuristics(contours, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    print "Stage2: Color Thresholding: ",time.time()-t," s"

    t=time.time()


    ######################## Stage 3

    #accepted, final_annotated_image = SVC(accepted,annotated_image,original_image,clf) # Removes false positives using a classifier
    accepted, final_annotated_image = NN(accepted,annotated_image,original_image, model) # Removes false positives using a classifier
    print "Stage3: Classifier: ",time.time()-t," s"

    t=time.time()

    # ######################## Finally
    # Total_Lights=len(BBs)
    # Recognized=0
    # for cnt in accepted:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     rguess = Rectangle(x, y, x+w, y+h)
    #     for bb in BBs:
    #         rtrue=Rectangle(bb)

    #         ratio = ret_area(rguess,rtrue)*1.0/((rtrue[2]-rtrue[0])*(rtrue[3])-(rtrue[1]))
    #         print ratio
    #         if ratio>0.5:
    #             Recognized+=1
                
    # print Recognized,Total_Lights



    # Visualize


    #print np.shape(gray_image)
    #print np.shape(heatmap)



    D=OrderedDict({"1) Original":cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),"2) Heatmap":heatmap,"3) Color Thresholded":thresh, "5) Annotated Image":final_annotated_image})


    #plot_figures(D,2,2)

    return time.time() - st, original_image,heatmap,thresh,final_annotated_image

