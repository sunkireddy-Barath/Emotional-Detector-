💤 Drowsiness Detection System

Author: Sunkireddy Barath
GitHub: https://github.com/sunkireddy-Barath

📌 Overview

The Drowsiness Detection System is a computer vision–based project designed to monitor a person’s alertness in real time.
The system analyzes facial features such as eye closure and yawning to detect signs of drowsiness and generate timely alerts.

This project is especially useful in driver monitoring systems, where early detection of drowsiness can help prevent accidents.
The project demonstrates the complete machine learning pipeline, including:

Data collection

Automatic labeling

Model training

Real-time detection and visualization

✨ Features

Real-Time Monitoring: Works with webcam or video input

Drowsiness Detection: Identifies eye closure and yawning patterns

Dual Model Approach: Separate models for eye state and yawning detection

Custom Dataset Support: Allows training using user-collected data

Auto Labeling: Reduces manual effort using automated bounding box generation

Live Visualization: Displays detection results and alerts in real time

📁 Project Structure

AutoLabelling.py – Automatically generates bounding boxes for training data

CaptureData.py – Captures video frames to build custom datasets

DrowsinessDetector.py – Main script for real-time detection and alerts

LoadData.ipynb – Loads and preprocesses datasets

RedirectData.ipynb – Organizes collected data into training format

train.ipynb – Trains the drowsiness detection models

⚙️ Installation
Clone the Repository
git clone https://github.com/sunkireddy-Barath/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System


(Update the repository name if different)

Create and Activate Virtual Environment
python -m venv venv


Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

▶️ How to Run

To start the real-time drowsiness detection system:

python DrowsinessDetector.py

🧪 Usage

Real-Time Detection:
Run DrowsinessDetector.py with a connected webcam

Data Collection:
Use CaptureData.py to collect images or video frames

Model Training:
Retrain models using train.ipynb with custom datasets

🧠 How the System Works

The system uses two object detection models:

👁 Eye State Detection

Detects whether eyes are open or closed

Helps identify prolonged eye closure, a key indicator of drowsiness

😮 Yawning Detection

Detects mouth open (yawning) or mouth closed

Used as an additional fatigue indicator

🔖 Auto Labeling

Bounding boxes are generated automatically

Improves dataset quality and training efficiency

The predictions from both models are combined and displayed in real time with visual indicators and alerts.

🛠 Technologies Used

Python

YOLOv8

OpenCV

TensorFlow / Keras

Auto-labeling tools

PyQt5 (for visualization)

⚠️ Important Note

This project is developed for academic and learning purposes to demonstrate the full workflow of a drowsiness detection system.
The provided model weights are for demonstration only and are not fully optimized for production environments.

🔮 Future Enhancements

Support for multiple person detection

Integration with wearable sensors

Mobile application deployment

Improved accuracy using larger and diverse datasets
