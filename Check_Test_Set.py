import pickle
import cv2
Test = pickle.load( open( "validation_set.p", "rb" ) )



for i in Test.keys():
	print i
	img=cv2.imread(i)

	for j in Test[i]:
		print j[0],j[1],j[2],j[3]
		cv2.rectangle(img, (j[0],j[1]), (j[2],j[3]),(0,255,0), 2)


	cv2.imshow("img",img)
	cv2.waitKey(1000)		