import cv2
import os
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a directory to save the face images if it doesn't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

id = input('İsim Giriniz: ')

sampleN = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sampleN += 1
        cv2.imwrite("faces/user." + str(id) + "." + str(sampleN) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if sampleN > 14:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()