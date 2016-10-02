import cv2,numpy as np

def NN(accepted_contours,annotated_image,original_image,model):
    classifier_results = [0]*len(accepted_contours)
    i = -1
    for cnt in accepted_contours:
        i += 1
        try:
            x,y,w,h = cv2.boundingRect(cnt)
            pad=int(0.125*w*h)
            temp=original_image[y:y+2*h,x-pad/3:x+w+pad/3]
            xx=cv2.resize(temp,(32,32))
            xx=np.asarray(xx)
            xx=np.transpose(xx,(2,0,1))
            xx=np.reshape(xx,(1,3,32,32))
            if model.predict_classes(xx,verbose=0)==[1]:
                cv2.rectangle(annotated_image, (x-pad/3,y-pad/3), (x+w+pad/3,y+h+pad/3),(255,100,10), 5)
                classifier_results[i] = 1
            else:
                classifier_results[i] = 0

        except:
            pass

    return classifier_results, annotated_image