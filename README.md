
# Smile Detector
This is a simple smile detector that uses the Multi-Task Cascaded Convolutional Neural Network (MTCNN) to detect faces and keypoints, and then extracts the face and labels it as smiling or not.

Installation
Use the package manager pip to install the required libraries:
```bash
  pip install mtcnn opencv-python numpy scikit-learn tensorflow
```

## Usage
To use the smile detector, you need to run the `smile_detection.py` file.

The script will go through all the images in the `smile_dataset` directory and its subdirectories, detect faces, and extract them. The extracted faces and their corresponding labels (smiling or not) will be saved in the `data` and `label` pickle files.

You can then use this data to train a smile detector model by uncommenting and running the last part of the `smile_detection.py` file, which uses TensorFlow to train a simple convolutional neural network (CNN) for classifying the extracted faces as smiling or not.

```bash
  python smile_detection.py
```
## Credits
The smile detector is based on the [MTCNN](https://github.com/ipazc/mtcnn) face detection library and the CNN classifier is based on the [TensorFlow](https://www.tensorflow.org/) machine learning library.
