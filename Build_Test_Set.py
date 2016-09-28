import cv2
import numpy as np
import random
import copy
import os
import pickle
# Build a validation set for testing and store it in a pickle file

print "Reading dataset..."








Images={}

dirs=os.listdir("dayTrain")

for d in dirs:
    f=open("dayTrain/"+d+"/"+"frameAnnotationsBULB.csv","r")
    raw=f.read()
    f.close()
    inputs=raw.split()
    inputs=[i.split(";") for i in inputs]
    for i in range(21):
        inputs.pop(0)
    for i in inputs:
        if i[1]=="stop":
            filename="dayTrain/"+d+"/frames/"+i[0][12:len(i[0])]
            tmp=cv2.imread(filename)
            print filename
            try:
                Images[filename].append([int(i[2]),int(i[3]),int(i[4]),int(i[5])])
            except:
                Images[filename]=[[int(i[2]),int(i[3]),int(i[4]),int(i[5])]]
            #print np.shape(Images[i[0]][0])
            #cv2.rectangle(tmp, (int(i[2]),int(i[3])), (int(i[4]),int(i[5])),(255,0,0), 2)
            #cv2.imshow("gg",tmp)
            #cv2.waitKey(1000)
        #if random.random()<=0.01:
        #    break



print "No of images: ",len(Images.keys())

Test={}

keys=random.sample( Images.keys(), 1000 )


for key in keys:
    Test[key]=Images[key]

pickle.dump( Test, open( "validation_set.p", "wb" ) )

