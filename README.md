# face-recognition

First try on face recognition program using Python, OpenCV and OpenCV's LBPH recognizer.

## Requirements

- You need Python 3.7. If you didn't install it yet, don't forget to add the environnement variable to `PATH` when the installer asks you!
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

1) Move your pictures (of human/animal/whatever faces) into `training_data` folder. The more you add faces of one single person, the more the recognition will be efficient.
You can of course remove all the sample data.

**Note** : If you want to identify anything other than frontal faces, you need to modify the `HAAR_CASCADE`. You can find other haar cascades [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).

2) For each person to identify, you need to rename his pictures like that : `<name><id>.jpg/png/...`. `id` starts by 1.

Your folder `training_data` should look like as follows :

```
training_data
├──morgan_freeman1.jpg
├──morgan_freeman2.jpg
├── ...
├──morgan_freeman12.jpg
├──uma_thurman1.jpg
├──...
├──uma_thurman16.png
```

### Test with a picture of your character

To test your pictures, there are two ways to do so :
- Pass it as an argument like `py main.py <url_of_your_image1> <url_of_your_image2> ... <url_of_your_imageX>` on command prompt
- Move your images into `test_data` folder, name is not important here

In any case, the program will display your image in a single window with the name of the person who it thinks it is (depending on your training data) and the confidence. The closer to 0 this number, the more accurate prediction is.

### Not satisfied with your results ?

Don't worry, there are many reasons that could explain your dissatisfaction.

#### `No face has been detected in this image`
If the program should notice a face but you receive this message, increase the value of the const variable `MIN_NEIGHBORS`. The more you increase it, the higher quality detection will get.

#### This is not what I was expected
If you get Morgan Freeman instead of Uma Thurman, you should probably add more images in the `training_data`.

## Credits

- [Url to image OpenCV converter](https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/) from pyimagesearch
- [Face recognition tutorial](https://www.superdatascience.com/blogs/opencv-face-recognition) from SuperDataScience
