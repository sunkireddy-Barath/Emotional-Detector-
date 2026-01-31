🚀 Machine Learning & Computer Vision Projects

Author: Sunkireddy Barath

This repository contains a collection of six machine learning and computer vision projects developed using Python.
Each project focuses on a real-world problem and demonstrates the use of deep learning, computer vision, and data processing techniques.

All models are implemented using custom datasets or publicly available datasets, trained using Jupyter Notebooks, and follow the given academic guidelines.

📌 Projects Overview

Drowsiness Detection Model

Nationality & Emotion Detection Model

Sign Language Detection Model

Car Colour Detection & Traffic Analysis Model

Attendance System with Emotion Detection

Animal Detection & Classification Model

1️⃣ Drowsiness Detection Model
📖 Description

This model detects whether a person is awake or sleeping in an image or video.
It can detect multiple people at the same time, identify who is sleeping, estimate their age, and display a pop-up alert showing the number of sleeping people.

Sleeping persons are highlighted with red bounding boxes.

⚙️ How It Works

Input image/video is captured.

Faces are detected using a face detection model.

Eye state is analyzed (open/closed).

If eyes are closed continuously → person is marked as sleeping.

Age is predicted using a pretrained age estimation model.

Sleeping people are counted and highlighted.

A pop-up displays:

Number of sleeping people

Their predicted ages

🧠 Models Used

Face Detection (YOLO / Haar Cascade)

Eye State Classification (CNN)

Age Prediction (Pretrained Deep Learning Model)

📂 Output

Red box → Sleeping person

Green/Blue box → Awake person

Pop-up message with count and age

▶️ How to Run
python drowsiness_detection.py

2️⃣ Nationality & Emotion Detection Model
📖 Description

This model predicts a person’s nationality and emotion from an image.
Based on nationality, additional attributes are predicted:

Nationality	Output
Indian	Age + Dress Color + Emotion
American	Age + Emotion
African	Dress Color + Emotion
Others	Nationality + Emotion
⚙️ How It Works

Face is detected from the uploaded image.

Facial features are analyzed using deep learning.

Nationality (ethnicity proxy) is predicted.

Emotion is detected using a CNN.

Conditional logic decides what extra attributes to show.

🧠 Models Used

Face Attribute Analysis Model

Emotion Detection CNN (FER-2013)

Color Detection (Histogram-based)

▶️ How to Run
python nationality_emotion.py

3️⃣ Sign Language Detection Model
📖 Description

This model recognizes hand signs and converts them into words or letters.
It works with images and real-time video and is active only between 6 PM and 10 PM.

⚙️ How It Works

Webcam or image input is captured.

Hand landmarks are detected using MediaPipe.

Landmark coordinates are passed to a trained model.

Gesture is classified into a known sign.

If current time is outside 6 PM – 10 PM, the model stops.

🧠 Models Used

MediaPipe Hands

LSTM / CNN Classifier

Custom Sign Language Dataset

▶️ How to Run
python sign_language.py

4️⃣ Car Colour Detection & Traffic Analysis Model
📖 Description

This model detects cars at a traffic signal, predicts their color, and counts the total number of cars.
It also detects people at the signal.

Blue cars → Red rectangle

Other color cars → Blue rectangle

⚙️ How It Works

Vehicles and people are detected using an object detection model.

Each detected car is cropped.

Car color is predicted using a trained CNN.

Bounding box color is decided based on car color.

Total cars and people are counted.

🧠 Models Used

YOLO Object Detection

Car Color Classification CNN

COCO Dataset (People Detection)

▶️ How to Run
python car_colour_detection.py

5️⃣ Attendance System with Emotion Detection
📖 Description

This system automatically marks student attendance using face recognition.
It also detects the emotion of each student and stores data in a CSV/Excel file with timestamp.

The system works only between 9:30 AM and 10:00 AM.

⚙️ How It Works

Student faces are trained beforehand.

Classroom image/video is captured.

Faces are recognized and matched.

If matched → student is marked Present

Emotion is detected for each student.

Attendance is saved with:

Name

Time

Emotion

Status

🧠 Models Used

Face Recognition (Deep Face Embeddings)

Emotion Detection CNN

CSV Logging System

📄 Output File Example
Name, Time, Emotion, Status
Barath, 09:41, Happy, Present

▶️ How to Run
python attendance_system.py

6️⃣ Animal Detection & Classification Model
📖 Description

This model detects and classifies multiple animals in images or videos.
It highlights carnivorous animals in red and shows a pop-up alert with the number of carnivores detected.

⚙️ How It Works

Image or video input is provided.

Animals are detected using an object detection model.

Each animal is classified by species.

If the animal is carnivorous:

Bounding box is red

Total carnivorous animals are counted.

Pop-up alert is displayed.

🧠 Models Used

YOLO Animal Detection

Custom Animal Classification Dataset

Rule-based Carnivore Identification

▶️ How to Run
python animal_detection.py

🛠 Technologies Used

Python

OpenCV

TensorFlow / PyTorch

MediaPipe

YOLO

NumPy, Pandas

Jupyter Notebook

📌 Conclusion

These projects demonstrate practical applications of machine learning and computer vision in real-world scenarios such as traffic monitoring, education, safety, and human–computer interaction.
Each model follows academic guidelines and focuses on clarity, correctness, and usability.
