FaceOff is a toy face recognition application built with [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/), OpenCV, and Tensorflow/Keras.
## Requirements
FaceOff requires Python 3.x but not greater than 3.6. This is important because Tensorflow doesn't support Python 3.7 yet.
## Setting Up The Python Environment
Having clonned this repo, it is recommended that you bootstrap a separate Python virtual environment to setup the codebase.
To create a virtual environment, you can either use Anaconda or the default venv module that ships with Python. Follow the 
instructions below to setup using either Anaconda, or venv.
## Setting Up With Anaconda
1. Download and install Anaconda from https://www.anaconda.com/
2. Run "conda env create -f environment.yml"
3. Run "conda activate faceoff"
## Setting Up With Python's Venv Module
1. CD into the base directory of this repository on your machine
2. Run "python3 -v venv env"
3. Run "source env/bin/activate"
4. Run "pip install -r requirements.txt"
## Downloading Pre-Trained VGG-Face Weights File
Because this application depends on the VGG-Face model to encode important facial features, you'll need to download the 
weights file. Click [here](https://drive.google.com/open?id=1e08UKnjof4aikx4zUqkxKMZ6OGbI-OMk) to download.
It's a big file (about 500MB and named vgg_face_weights.h5) so be patient.
After downloading, copy the weights file into the root of the codebase on your machine.
## Running The Application
In the root of the codebase, fun the following command:*

**python3 main.py --weights vgg_face_weights.h5 --distance 0.2**
The flag **--weights** specifies the path to the pre-trained VGG-Face weights file and **--distance** specifies the 
similarity threshold used to determine matched faces. The value for **--distance** MUST ALWAYS be a number between 0 and 1
since the application uses the Cosine Similarity Measure.

You can see a video of it in action on my [AI Weekend Hacks playlist](https://www.youtube.com/watch?v=GBMTg9uHRr4)

## How Does This Work
1. OpenCV's HaarsCascade classifier is used to detect faces from the camera feed.
2. Rectangles are drawn where the classifier sees faces
3. Pressing the 'r' key will let your register a new user by entering his/her fullname in the terninal
4. Whiles registering a new user, make sure he/she is the only face seen by the camera
5. VGG-Face is used to encode facial features into 2622-dimensional vectors and store in /faces as [username].npy
6. When multiple faces are identified by the classifier, matching is done by comparing cosine distance of all known faces.
7. When matches are found, their names are shown. Otherwise, "unknown user" is shown.

## Do You Want To Improve This?
You're definitely welcome to submit pull requests for improvements. Below are some of the immediate improvements that can be made:
1. Use OpenCV's DNN classifer instead of HaarsCascade classifier.
2. Use DLib to detect faces from varied angles instead of HaarsCascade which doesn't really work well.
3. Improve performance
4. Improve documentation.
