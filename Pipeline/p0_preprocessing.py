import time, cv2, copy, numpy as np

def preprocess(image):
	start=time.time()
        img=copy.copy(image)
        height, width, channels = img.shape 
	
        #img = cv2.medianBlur(img,3) # Median Blur to Remove Noise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        b,g,r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8)) # Adaptive histogram equilization
        clahe = clahe.apply(r)
        img = cv2.merge((b,g,clahe))
	print "Preprocessing: " , time.time() - start, " ms" 
        return image,img




