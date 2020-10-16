import numpy as np
import cv2
import pickle

#import cascade file
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#get the recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {} #created a dictionary for add labels

#load label from pickle
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}



#add trained data
recognizer.read("trainner.yml")

#VideoCapture can be used to snap pictures
cap  = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read() #read frame by frame

    #convert frame into gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #get face from camera image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        #region of interest
        #cordinates for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #recognize
        #opencv has deep learn model predict keras tensortflow pytorch

        id_,conf = recognizer.predict(roi_gray)

        if conf >= 45: # and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)


        img_item = "my-image.png"
        #save just the potion of face
        cv2.imwrite(img_item,roi_gray)

        #draw a rectangle
        color = (255,0,0) #BGR 0-255
        stroke = 2 #border weight  
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y),color,stroke)

        

    cv2.imshow('frame',frame) #display frame coming from cap.read()

    #if not camera will crash
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done,release the camera
cap.release()
cv2.destroyAllWindows()