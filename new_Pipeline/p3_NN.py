import cv2,numpy as np

def NN(accepted_contours,annotated_image,original_image,model):
    classifier_results = [0]*len(accepted_contours)
    i = -1
    for cnt in accepted_contours:
        i += 1
        try:
            x,y,w,h = cv2.boundingRect(cnt)
            center=(x+(w/2.0),y+(h/2.0))
            r_x1=x-50
            r_y1=y-50
            r_x2=x+50
            r_y2=y+50

            temp=original_image[r_y1:r_y2,r_x1:r_x2]
            #cv2.imshow("img",temp)
            #cv2.waitKey(10000)
            xx=cv2.resize(temp,(64,64))
            xx=np.asarray(xx)
            xx=np.transpose(xx,(2,0,1))
            xx=np.reshape(xx,(1,3,64,64))
            if model.predict_classes(xx,verbose=0)==[1]:
                cv2.rectangle(annotated_image, (r_x1,r_y1), (r_x2,r_y2),(255,100,10), 5)
                classifier_results[i] = 1
            else:
                classifier_results[i] = 0

        except:
            classifier_results[i] = 1 # Handle this Later
    return classifier_results, annotated_image