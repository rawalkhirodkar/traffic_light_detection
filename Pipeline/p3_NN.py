import cv2,numpy as np

def NN(contours,annotated_image,original_image,model):
    accepted=[]
    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)
        pad=int(0.125*w*h)
        temp=original_image[y:y+2*h,x-pad/3:x+h+pad/3]
        xx=cv2.resize(temp,(32,32))
        xx=np.asarray(xx)
        xx=np.transpose(xx,(2,0,1))
        xx=np.reshape(xx,(1,3,32,32))
        if model.predict_classes(xx,verbose=0)==[1]:
            cv2.rectangle(annotated_image, (x-pad,y-pad), (x+w+pad,y+h+pad),(255,100,10), 5)
            accepted.append(cnt)


    return accepted, annotated_image