import numpy as np
import cv2

#VideoCapture can be used to snap pictures
cap  = cv2.VideoCapture(0)

#setting resolution
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while True:
    ret, frame = cap.read() #read frame by frame
    frame75 = rescale_frame(frame, percent=75)
    cv2.imshow('frame75',frame75) #display frame coming from cap.read()
    
    frame150 = rescale_frame(frame, percent=150)
    cv2.imshow('frame150', frame150)

    #if not camera will crash
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done,release the camera
cap.release()
cv2.destroyAllWindows()