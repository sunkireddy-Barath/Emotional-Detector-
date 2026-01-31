✋ Sign Language Recognition using MediaPipe & MLP

GitHub: https://github.com/sunkireddy-Barath

📌 Project Overview

This project focuses on real-time sign language recognition using computer vision and deep learning.
The system detects hand gestures from a webcam, extracts hand landmark features using MediaPipe Hands, and classifies sign language characters using a Multi-Layer Perceptron (MLP) model.

During development, two approaches were explored:

MobileNetV2 (image-based classification)

MediaPipe + MLP (landmark-based classification)

After experimentation, the MediaPipe-based MLP approach performed better for real-time recognition, offering higher speed and stability.

The system dynamically converts detected signs into letters, which can form words and sentences.

📂 Project Structure

The project is organized as follows:

Sign Language Recognition System

SIGN_TO_SENTENCE_PROJECT/

Asl_Sign_Data/ – Raw ASL sign language dataset

asl_mediapipe_keypoints_dataset.csv – Landmark-based dataset for MLP training

asl_mediapipe_mlp_model.h5 – Trained MLP model

sign_language_model_MobileNetV2.h5 – Trained MobileNetV2 model

Combined_Architecture.ipynb – Hybrid experiment notebook

LLM.ipynb – Language model integration experiments

Mediapipe_Training.ipynb – Training notebook for MediaPipe + MLP

MobileNetV2_Training.ipynb – Training notebook for MobileNetV2

conclusion.txt – Summary of observations and results

requirements.txt – Required dependencies

🏗️ Dataset

The dataset used for training is based on the ASL (American Sign Language) dataset from Kaggle.

Contains hand gesture images labeled with ASL characters

MobileNetV2 was trained using raw images

MediaPipe + MLP was trained using extracted hand landmark keypoints

Landmark data was converted into CSV format for efficient training

🧠 Model Approaches
1️⃣ MobileNetV2 (Image-Based Approach)

A pre-trained MobileNetV2 model was fine-tuned for sign classification

Works well on static images

Performance drops in real-time webcam scenarios due to lighting and motion variations

Conclusion:
Not ideal for real-time sign language recognition in this use case.

2️⃣ MediaPipe + MLP (Landmark-Based Approach)

Hand landmarks are extracted using MediaPipe Hands

Landmark coordinates are flattened and passed to an MLP classifier

Faster, more reliable, and better suited for real-time recognition

Final Choice:
✅ MediaPipe + MLP model

🚀 How to Run the Project
Step 1: Install Dependencies

Install all required libraries using:

pip install -r requirements.txt

Step 2: Model Evaluation

To evaluate trained models:

Run the last cell in Mediapipe_Training.ipynb for the MLP model

Run the last cell in MobileNetV2_Training.ipynb for the MobileNetV2 model

Step 3: Run Combined Architecture

To test both approaches together:

jupyter notebook Combined_Architecture.ipynb

🎮 Controls & Gesture Logic

Normal Signs → Letters are added to the sentence

SPACE Sign → Adds a space

DELETE Sign → Removes the last character

NOTHING → No action is performed

This logic allows the system to form complete words and sentences dynamically.

⚠️ Limitations

MobileNetV2 struggles with real-time webcam input

System currently supports a limited set of predefined signs

Accuracy depends on lighting conditions and camera quality

🔮 Future Enhancements

FastAPI backend integration for mobile and web apps

Text-to-Sign Language generation using:

AI-based 3D avatars

Deep learning–based gesture synthesis

Improved sentence-level understanding using language models

Support for additional sign languages

🤝 Acknowledgments

MediaPipe Hands for real-time hand landmark detection

TensorFlow and Scikit-Learn for model development

Research work in gesture recognition and sign language AI

📜 License

This project is licensed under the MIT License.
Refer to the LICENSE file for more details.
