# 🚀 Machine Learning & Computer Vision Projects

**Author:** Sunkireddy Barath  

This repository contains **six machine learning and computer vision projects** implemented using Python.  
Each project solves a **real-world problem** using deep learning and computer vision techniques.

---

## 📌 Projects Included
1. Drowsiness Detection  
2. Nationality & Emotion Detection  
3. Sign Language Detection  
4. Car Colour Detection & Traffic Analysis  
5. Attendance System with Emotion Detection  
6. Animal Detection & Classification  

---

## 1️⃣ Drowsiness Detection Model
**Description:**  
Detects whether a person is **awake or sleeping** in images/videos. Supports **multiple people**, predicts **age**, and shows a pop-up with the count of sleeping people.

**How it works:**
- Detect faces
- Analyze eye state (open/closed)
- Predict age
- Mark sleeping persons in **red**

**Models Used:** Face Detection, CNN (Eye State), Age Prediction  
**Run:**  
```bash
python drowsiness_detection.py
2️⃣ Nationality & Emotion Detection Model
Description:
Predicts nationality and emotion from an image. Outputs vary based on nationality.

How it works:

Face detection

Nationality & emotion prediction

Rule-based attribute display

Models Used: Face Attributes, Emotion CNN
Run:

python nationality_emotion.py
3️⃣ Sign Language Detection Model
Description:
Recognizes hand gestures and converts them to words using images or real-time video. Active only from 6 PM – 10 PM.

How it works:

Detect hand landmarks

Classify gestures

Time-based execution control

Models Used: MediaPipe Hands, LSTM/CNN
Run:

python sign_language.py
4️⃣ Car Colour Detection & Traffic Analysis Model
Description:
Detects cars and people at traffic signals. Predicts car color and counts vehicles.

Blue car → Red box

Other cars → Blue box

How it works:

Vehicle & person detection

Car color classification

Object counting

Models Used: YOLO, Car Color CNN
Run:

python car_colour_detection.py
5️⃣ Attendance System with Emotion Detection
Description:
Automatically marks student attendance using face recognition and detects emotions. Works only from 9:30 AM – 10:00 AM.

How it works:

Face recognition

Emotion detection

CSV/Excel attendance logging

Models Used: Face Recognition, Emotion CNN
Run:

python attendance_system.py
6️⃣ Animal Detection & Classification Model
Description:
Detects and classifies multiple animals in images/videos. Carnivorous animals are highlighted in red with a pop-up alert.

How it works:

Animal detection

Species classification

Carnivore count alert

Models Used: YOLO, Animal Classification CNN
Run:

python animal_detection.py
🛠 Technologies Used
Python

OpenCV

TensorFlow / PyTorch

YOLO

MediaPipe

NumPy, Pandas

Jupyter Notebook

