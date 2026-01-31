🚗 Car Color Classification Neural Network

GitHub: https://github.com/sunkireddy-Barath

📌 Executive Summary

This project focuses on building a Car Color Classification system using a Convolutional Neural Network (CNN).
The model is trained to predict the dominant color of a car from an image using deep learning techniques.

The system is developed using TensorFlow and Keras, and car images are sourced from the Stanford Cars Dataset.
Each car image is analyzed in the RGB color space, and the closest matching color is predicted based on dominant color values.

To improve performance, image augmentation techniques are applied during training.
The overall objective is to accurately map a car image to its most suitable color category.

📖 Background

Vehicle colors play an important role in applications such as traffic monitoring, surveillance, insurance verification, and smart transportation systems.
Car manufacturers use specific color codes, but from an image perspective, color can be represented using RGB values.

By training a CNN model on car images, it becomes possible to predict the closest RGB-based color from a given car image.
This project demonstrates how deep learning can be used to automate color recognition from visual data.

🧠 Model Overview

A CNN-based architecture is used for car color classification

Input images are processed in RGB format

The dominant color of each car image is calculated

Color classes are generated using RGB distance comparison

The model predicts the closest matching color class

📊 Dataset Details

Dataset Used: Stanford Cars Dataset

Training Images: 6,108

Validation Images: 2,036

Prediction Images: 8,041

Total Color Classes: 200

Total Model Parameters: ~44 million

Each image is labeled based on its dominant RGB color and mapped to the nearest predefined color category.

🏗️ Model Architecture

Convolutional layers for feature extraction

Pooling layers for dimensionality reduction

Fully connected layers for classification

Softmax output layer for multi-class color prediction

The architecture is inspired by research work on vehicle color recognition using CNNs.

📈 Results & Observations

Validation Accuracy: 16.5%

Baseline Accuracy: 0.5%

Although the accuracy is significantly higher than the baseline, performance is limited due to the large number of color classes.

🧪 Challenges

Too many color classes make classification difficult

Similar colors often overlap in RGB space

Lighting conditions affect color perception

🔮 Future Improvements

Reduce the number of color classes

Group similar colors together

Apply stronger data augmentation

Fine-tune the CNN architecture

Integrate the model with a traffic analysis system

🛠 Technologies Used

Python

TensorFlow

Keras

OpenCV

Convolutional Neural Networks (CNN)

📚 References

Stanford Cars Dataset:
https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Vehicle Color Recognition Using CNN:
Reza Fuad Rachmadi, I Ketut Eddy Purnama
https://arxiv.org/pdf/1510.07391.pdf

RGB Color Dataset:
https://github.com/codebrainz/color-names
