#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ananya Srinivasan
CSC 355: Human Computer Interaction
Homework 4: Computer Vision, Exercise 2
Description: This program utilizes Haar Cascading to detect facial and eye movements. The coordinates of the facial and eye 
movements are extracted and compared to determine whether a person in a video is nodding "yes" or "no." 

"""
import cv2

#Get classifiers for eyes and face
#Reference: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
face_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('./haarcascade_eye.xml') 

#Get video and resize
cap = cv2.VideoCapture('./yes-no.avi')
cap.set(3,640)
cap.set(4,480)

#Set up at table of the eye and face movements - history
history = []
counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if(len(eyes) >= 2):
            history.append((x, y, eyes[0][0], eyes[0][1])) #Left eye
            history.append((x, y, eyes[1][0], eyes[1][1])) #Right eye
    
    #Compare to previous head movement and determine whether the position is a "yes" or "no"
    if counter < len(history)-1:
        if abs(history[counter+1][0] - history[counter][0]) > abs(history[counter+1][1] - history[counter][1]):
            #more x movement than y movement
            print("NO")
        elif abs(history[counter+1][1] - history[counter][1]) > abs(history[counter+1][0] - history[counter][0]):
            #more y movement than x movement
            print("YES")
    counter += 1
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break     
            
cap.release()
cv2.destroyAllWindows()
