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
        print("Error : No image in this folder")
        return

    for image in images:
        face = detect_face(folder_path + "/" + image)[0]
        if face is not None:
            faces.append(face)
            labels.append(re.sub(r"\d.\w*","",image))

    # Convert labels into numbers (0 = first character, 1 = second...)
    guinea_pigs = np.unique(labels)
    for i in range(0,len(labels)):
        if labels[i] in guinea_pigs:
            index, = np.where(guinea_pigs == labels[i])
            labels[i] = index
    return faces, guinea_pigs, labels

def detect_face(my_image):
    img = cv.imread(my_image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    x, y, w, h = faces[0]
    return gray[y:y+h,x:x+w], faces[0]

def train_data(faces, labels):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

faces, guinea_pigs, labels = prepare_data(TRAINING_DATA_PATH)
face_recognizer = train_data(faces, labels)

def who_is_this(my_image):
    face, rect = detect_face(my_image)

    img = cv.imread(my_image)
    label = face_recognizer.predict(face)[0]
    label_text = guinea_pigs[label]
    print(label_text)
    
    x, y, w, h = rect
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.putText(img, label_text, (x, y-5), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
    
    cv.imshow('Press any key to exit...',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Debug
who_is_this(TRAINING_DATA_PATH + "/uma_thurman1.jpg")