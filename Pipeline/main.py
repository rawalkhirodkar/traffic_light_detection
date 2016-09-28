from p0_preprocessing import *
from p1_VGG import *
from p2_colorthresholding import *
from collections import *
import cv2
import numpy as np
import matplotlib.pyplot as plt



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



# Initialization

filename="test.jpg"
ids,model_heatmap = initialize_vgg()

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


# Visualize


#print np.shape(gray_image)
#print np.shape(heatmap)



D=OrderedDict({"1) Original":cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),"2) Heatmap":heatmap,"3) Color Thresholded":thresh, "5) Annotated Image":annotated_image})


plot_figures(D,2,2)

