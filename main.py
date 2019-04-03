import os
import re
import numpy as np
import cv2 as cv

TRAINING_DATA_PATH = "training_data"
HAAR_CASCADE = "haarcascade_frontalface_default.xml"

def prepare_data(folder_path):
    faces = []
    labels = []
    images = os.listdir(folder_path)

    if len(images) == 0:
        print("No image in this folder")
        return

    for image in images:
        face = detect_face(folder_path + "/" + image)
        if face is not None:
            faces.append(face)
            labels.append(re.sub(r"\d.\w*","",image))
    return faces, labels

def detect_face(my_image):
    img = cv.imread(my_image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    x, y, w, h = faces[0]
    return gray[y:y+h,x:x+w]

faces, labels = prepare_data(TRAINING_DATA_PATH)