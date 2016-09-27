from p0_preprocessing import *
from p1_VGG import *
import cv2
import numpy as np

original_image, processed_image=preprocess(cv2.imread("test.jpg"))
ids,model_heatmap=initialize_vgg()









cv2.imshow("Visualization",np.append(np.asarray(original_image),np.asarray(processed_image),axis=1))
cv2.waitKey(0)
