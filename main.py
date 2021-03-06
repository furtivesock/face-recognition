import os
import re
import numpy as np
import cv2 as cv
import sys
import urllib.request as urllib

# Const variables
TRAINING_DATA_PATH = "training_data"
TEST_DATA_PATH = "test_data"
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 8

def url_to_image(url):
    """Convert the url into an image.
    Credits to pyimagesearch
    """
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image

# Images urls (separated by spaces) passed as an argument to the program, if not, it will take all images contained in test_data folder
images_to_predict = []
if len(sys.argv) > 1:
    for i in range(1,len(sys.argv)):
        url = sys.argv[i]
        images_to_predict.append(url_to_image(url))
else:
    for image in os.listdir(TEST_DATA_PATH):
        images_to_predict.append(TEST_DATA_PATH + "/" + image)
    print(images_to_predict)
    if len(images_to_predict) == 0:
        sys.exit("Error : No image has been found in the test_data folder")

def get_image(my_image):
    """Return an image (numpy array)
    """

    if isinstance(my_image,np.ndarray):
        return my_image
    else:
        return cv.imread(my_image)

def prepare_data(folder_path):
    """Prepare the labeled images in training_data folder
    to make them usable for opencv functions
    Return a list of faces, a list of our guinea pigs names and a list of labels (integer)
    """

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
    """Detect face from a given image
    Return the face part and the cropping coordinates
    """

    img = get_image(my_image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS)
    
    # If no face is detected
    if (len(faces) == 0):
        return None, None

    x, y, w, h = faces[0]
    return gray[y:y+h,x:x+w], faces[0]

def train_data(faces, labels):
    """Train the face recognizer of our faces
    Return the LBPH face recognizer
    """

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

print("Preparing data...")
faces, guinea_pigs, labels = prepare_data(TRAINING_DATA_PATH)
print("Training the", len(faces), "faces...")
face_recognizer = train_data(faces, labels)

def who_is_this(my_image):
    """Predict who owns this face
    according to our training data
    Show the image with the name of the mystery person and its associated confidence
    """

    img = get_image(my_image)
    face, rect = detect_face(img)

    # If no face detected
    if face is None:
        print("No face has been detected in this image")
        return

    label, confidence = face_recognizer.predict(face)
    label_text = guinea_pigs[label] + " " + str(confidence)
    
    print("Prediction :", label_text)

    x, y, w, h = rect
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 4)
    cv.putText(img, label_text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    height, width = img.shape[:2]
    resized_img = cv.resize(img, (int(0.6*width), int(0.6*height)))
    cv.imshow("Press any key to exit...",resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

print("Predicting your data...")
for image in images_to_predict:
    who_is_this(image)

sys.exit()