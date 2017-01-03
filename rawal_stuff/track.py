
import os
import glob
import dlib
import cv2
from skimage import io

# Path to the video frames
video_folder = os.path.join("..", "examples", "video_frames")

# Create the correlation tracker - the object needs to be initialized
# before it can be used
tracker = dlib.correlation_tracker()
face_detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

k = 0
track_begin = False

cap=cv2.VideoCapture("../videos/new_1.avi")
while(cap.isOpened()):
    ret, frame = cap.read()
    dets = face_detector( frame, 0)
    for i, d in enumerate( dets ):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
                screenArea = (d.right() - d.left()) * (d.bottom() - d.top())
                cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
                if(i == 0):
                    tracker.start_track(frame, dlib.rectangle(d.left(), d.top(), d.right(), d.bottom()))
                    track_begin = True
    
    if track_begin == True:
        # Else we just attempt to track from the previous frame
        tracker.update(frame)
        rect = tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
    
    cv2.imshow('feed',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
    k+=1