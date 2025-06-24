---

## Features

- Capture and store user face images
- Train a face recognizer using the LBPH algorithm
- Real-time face detection and recognition using webcam
- Unlocks a screen via browser if the face is recognized
- Easy to run locally

---

## Tech Stack

- Python 3.13
- OpenCV (contrib version)
- Flask
- HTML & JavaScript

---

## Project Structure

face_lock/
├── capture_faces.py
├── train_model.py
├── face_lock.py
├── index.html
├── dataset/
├── trainer.yml
├── .gitignore
└── README.md


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/manas81622/face_lock.git
cd face_lock

# Set up Virtual Environment #

python -m venv venv
source venv/bin/activate   # Mac/Linux
# .\venv\Scripts\activate  # Windows

# Install Dependencies #

pip install opencv-contrib-python flask

# Capture Face Data #

python capture_faces.py

# Train the Face Recognition Model #

python train_model.py

# Run the Flask Server #

python face_lock.py
