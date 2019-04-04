import os
import re
import numpy as np
import cv2 as cv
import sys
import urllib.request as urllib

TRAINING_DATA_PATH = "training_data"
HAAR_CASCADE = "haarcascade_frontalface_default.xml"

# If you don't want to pass argument to the program, you can replace the url below instead :
URL_TO_TEST_IMAGE = "https://framapic.org/qpcBEgtWh6o5/NysVt07uOiPc.jpg"

# Convert the url into an image. Credits to pyimagesearch
def url_to_image(url):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv.imdecode(image, cv.IMREAD_COLOR)
	return image

# Url passed as an argument to the program
if len(sys.argv) == 1:
    url = URL_TO_TEST_IMAGE
else:
    url = sys.argv[1]
image_to_predict = url_to_image(url)

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
    if isinstance(my_image,np.ndarray):
        img = my_image
    else:
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

    if isinstance(my_image,np.ndarray):
        img = my_image
    else:
        img = cv.imread(my_image)
    label, confidence = face_recognizer.predict(face)
    label_text = guinea_pigs[label] + " " + str(confidence)
    
    x, y, w, h = rect
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 4)
    cv.putText(img, label_text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    height, width = img.shape[:2]
    resized_img = cv.resize(img, (int(0.5*width), int(0.5*height)))
    cv.imshow("Press any key to exit...",resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

who_is_this(image_to_predict)