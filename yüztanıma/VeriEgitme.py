import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the face images and labels
face_images = []
face_labels = []

for dirname, _, filenames in os.walk('faces'):
    for filename in filenames:
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(dirname, filename))
            face_images.append(img)
            face_labels.append(int(os.path.splitext(filename)[0].replace('user', '')))

# Train the recognizer with the face images and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_images, np.array(face_labels))

# Save the trained recognizer to a file
recognizer.save('recognizer.yml')