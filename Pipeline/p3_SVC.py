import cv2,numpy as np

def SVC(contours,annotated_image,original_image,clf):
    accepted=[]
    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)
        pad=int(0.125*w*h)
        temp=original_image[y:y+2*h,x-pad/3:x+h+pad/3]
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        preds=clf.predict([np.ndarray.flatten(cv2.resize(temp,(64,64)))])

 
        if preds[0]==1:
            cv2.rectangle(annotated_image, (x-pad,y-pad), (x+w+pad,y+h+pad),(255,100,10), 5)
            accepted.append(cnt)


    return accepted, annotated_image