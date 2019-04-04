# face-recognition

First try on face recognition algorithm for my English class

## Requirements

- You need Python 3.7. If you didn't install it yet, don't forget to add the environnement variable to PATH when the installer asks you!
- Type `pip install opencv-python==3.4.2.16`. OpenCV is a library with many and useful features for face detection and recognition.
- Type `pip install opencv-contrib-python==3.4.2.16` to use LBPH face recognizer, the algorithm used for training and recognizing your guinea pigs.
- Type `pip install numpy`. We will need its N-dimensional arrays for OpenCV face recognizer.

## Installation

*Clone* the repository :
```
git clone https://github.com/furtivesock/face-recognition.git
cd face-recognition
```

## Usage

### Feed the program with your faces

1) Move your pictures (of human/animal/whatever faces) into `/training_data` folder. The more you add faces of one single person, the more the recognition will be efficient.
2) For each person to identify, you need to rename his pictures like that : `[name][id].jpg/png/...`. `id` starts by 1.

Your folder /training_data should look like as follows :

```
training_data
├──alerick1.jpg
├──alerick2.jpg
├── ...
├──alerick12.jpg
├──delphine1.jpg
├──...
├──delphine16.png
```

### Test with a picture of your character

To test your picture, you need to get its url on the Internet (I made this choice to simplify my tests). Then, you have two ways to test it :
- Pass it as an argument like `py main.py [url_of_your_image]`
- Or, you can modify the const variable `URL_TO_TEST_IMAGE` at the beginning of the code. It must be a string.

In any case, the program will display your image in a single window with the name of the person who it thinks it is (depending on your training data) and the confidence. The closer to 0 this number, the more accurate prediction is.