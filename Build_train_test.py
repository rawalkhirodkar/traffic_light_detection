import pickle
import cv2
import collections
import random

# Pickles into Train/Test split, 80:20 ratio
# Also find the largest rectangle

Data = pickle.load( open( "full_training_set_links.p", "rb" ) )

Train = {}
Test = {}
maxi=0
for i in Data.keys():
	if random.random()>0.2:
		Train[i]=Data[i]
	else:
		Test[i]=Data[i]
	rect=Data[i][0]
	if (rect[2]-rect[0]) * (rect[3]-rect[1])>maxi:
		maxi=(rect[2]-rect[0]) * (rect[3]-rect[1])
		max_img=i
		max_rect=Data[i]






print "Maximum Rectangle Size", maxi
j=max_rect[0]
img=cv2.imread(i)
#cv2.rectangle(img, (j[0],j[1]), (j[2],j[3]),(0,255,0), 2)


for j in Data["dayTrain/dayClip9/frames/dayClip9--00497.png"]:
		print j[0],j[1],j[2],j[3]
		print (j[2]-j[0]) * (j[3] - j[1])
		cv2.rectangle(img, (j[0],j[1]), (j[2],j[3]),(0,255,0), 2)


cv2.imshow("img",img)
cv2.waitKey(0)	
print "Training Size", len(Train)
print "Testing",len(Test)


'''
# Data = collections.OrderedDict(sorted(Data.items()))

for i in Data.keys():
	print i
	img=cv2.imread(i)

	for j in Data[i]:
		print j[0],j[1],j[2],j[3]
		cv2.rectangle(img, (j[0],j[1]), (j[2],j[3]),(0,255,0), 2)


	cv2.imshow("img",img)
	cv2.waitKey(1000)		
'''


pickle.dump( Train, open( "training_links.p", "wb" ) )
pickle.dump( Test, open( "testing_links.p", "wb" ) )