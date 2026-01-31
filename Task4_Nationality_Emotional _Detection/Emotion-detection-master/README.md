😊 Emotion Detection Using Deep Learning

GitHub: https://github.com/sunkireddy-Barath

Introduction

This project implements a facial emotion detection system using deep learning.
The goal is to identify a person’s emotional state from facial expressions in images or real-time webcam input.

The system classifies emotions into seven categories:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

The model is trained using the FER-2013 dataset, which consists of grayscale facial images of size 48×48 pixels.
Emotion detection plays an important role in applications such as smart classrooms, attendance systems, human–computer interaction, and surveillance systems.

Dependencies

The following tools and libraries are required to run this project:

Python 3

OpenCV

TensorFlow (Keras API)

All dependencies can be installed using the provided requirements.txt file.

Basic Usage

This project is developed using TensorFlow 2.x and the Keras deep learning framework.

Step 1: Clone My Repository
git clone https://github.com/sunkireddy-Barath/Emotion-Detection.git
cd Emotion-Detection


(Replace the repository name if your folder name is different)

Step 2: Dataset Setup

Download the FER-2013 dataset and place it inside the src directory.

Step 3: Train the Model

To train the emotion detection model:

cd src
python emotions.py --mode train

Step 4: Run Emotion Detection

To run emotion detection using the trained or saved model:

cd src
python emotions.py --mode display


The system will open the webcam, detect faces, and display the predicted emotion for each detected face.

Project Structure

The project directory is organized as follows:

src/data/ – Emotion dataset

emotions.py – Main script for training and emotion prediction

haarcascade_frontalface_default.xml – Face detection model

model.h5 – Trained CNN emotion detection model

Model Performance

Uses a 4-layer Convolutional Neural Network (CNN)

Achieves around 63% accuracy after 50 training epochs

Designed to be lightweight and suitable for real-time emotion detection

Data Preparation (Optional)

The original FER-2013 dataset is provided as a CSV file.
For easier processing, the dataset is converted into image format (PNG).

A helper script is included to:

Convert CSV data into images

Separate training and testing data

This allows experimentation with custom datasets in the future.

Algorithm Explanation

Faces are detected from the webcam using Haar Cascade Classifier

The detected face region is resized to 48×48 pixels

The image is passed to the CNN model

The model outputs probability scores for each emotion

The emotion with the highest probability is displayed on the screen

Technologies Used

Python

OpenCV

TensorFlow / Keras

Haar Cascade Face Detection

Convolutional Neural Networks (CNN)

Conclusion

This project demonstrates the use of deep learning for real-time facial emotion recognition.
It serves as a strong base for integrating emotion detection into larger systems such as attendance monitoring, student behavior analysis, and intelligent surveillance applications.
