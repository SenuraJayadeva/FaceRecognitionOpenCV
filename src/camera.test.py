import numpy as np
import cv2

#VideoCapture can be used to snap pictures
cap  = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read() #read frame by frame

    cv2.imshow('frame',frame) #display frame coming from cap.read()

    #if not camera will crash
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done,release the camera
cap.release()
cv2.destroyAllWindows()