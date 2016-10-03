import os 
import cv2
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC


pos_list=os.listdir("pos")[0:2000]
neg_list=os.listdir("neg")[0:6000]

#print pos_list

#cv2.imshow("d",cv2.imread("pos/"+pos_list[1],0))
#cv2.waitKey(0)


pos_list=[np.ndarray.flatten(cv2.resize(cv2.imread("pos/"+i,0),(64,64))) for i in pos_list]
neg_list=[np.ndarray.flatten(cv2.resize(cv2.imread("neg/"+i,0),(64,64))) for i in neg_list]
#print np.shape(pos_list)



X=pos_list+neg_list
y=[1]*len(pos_list)+[0]*len(neg_list)

print np.shape(X)
print np.shape(y)



clf =  SVC(kernel='poly', C=30, degree=2, max_iter=5000000)

scores = cross_validation.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf.fit(X,y)

import cPickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    