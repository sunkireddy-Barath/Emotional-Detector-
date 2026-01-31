Drowsiness Detection System

GitHub: https://github.com/sunkireddy-Barath

Overview

The Drowsiness Detection System is a computer vision–based project developed to monitor a person’s alertness in real time.
The system analyzes facial features such as eye closure and yawning to detect signs of drowsiness and provide timely alerts.

This project is especially useful for driver monitoring systems, where detecting drowsiness early can help reduce accidents.
The repository demonstrates the complete development workflow, including data collection, automatic labeling, model training, and real-time detection.

Features

Real-time Monitoring: Works with webcam or video input

Drowsiness Detection: Identifies eye closure and yawning patterns

Dual Model Approach: Separate models for eye state and yawning detection

Custom Dataset Support: Allows collecting and training on personal datasets

Auto Labeling: Reduces manual work using automated bounding box generation

Live Visualization: Displays detection results and alerts in real time

Project Structure

AutoLabelling.py – Automatically generates bounding boxes for training data

CaptureData.py – Captures video frames for building custom datasets

DrowsinessDetector.py – Main script for real-time detection and alerts

LoadData.ipynb – Loads and preprocesses datasets

RedirectData.ipynb – Organizes collected data into training format

train.ipynb – Trains the drowsiness detection models

Installation
Clone the Repository
git clone https://github.com/tyrerodr/Real_time_drowsy_driving_detection.git
cd Real_time_drowsy_driving_detection

Create and Activate Virtual Environment
python -m venv venv


Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

How to Run

To start the real-time drowsiness detection system, run:

python DrowsinessDetector.py

Usage

Real-Time Detection: Run the main script with a connected webcam

Data Collection: Use CaptureData.py to collect images or video frames

Model Training: Retrain the models using train.ipynb with custom datasets

How the System Works

The system uses two object detection models:

Eye State Detection

Detects whether eyes are open or closed

Helps identify prolonged eye closure, a key sign of drowsiness

Yawning Detection

Detects mouth open (yawning) or mouth closed

Used as an additional indicator of fatigue

Auto Labeling

Automatic bounding box generation is used to prepare datasets

Improves training efficiency and dataset quality

The predictions from both models are combined and displayed in real time with visual indicators and alerts.

Technologies Used

Python

YOLOv8

OpenCV

TensorFlow / Keras

Auto-labeling tools

PyQt5 (for visualization)

Important Note

This project is developed for academic and learning purposes to demonstrate the complete pipeline of a drowsiness detection system.
The provided model weights are for demonstration only and are not fully optimized for production use.

Future Enhancements

Support for multiple person detection

Integration with wearable sensors

Mobile application deployment

Improved accuracy with larger and more diverse datasets
