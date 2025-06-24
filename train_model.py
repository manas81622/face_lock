import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    face_samples = []
    ids = []
    for image_path in image_paths:
        id = int(os.path.split(image_path)[-1].split(".")[0])  # Extract the count part
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(gray_img)
        for (x, y, w, h) in faces:
            face_samples.append(gray_img[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids



faces, ids = get_images_and_labels('dataset')
recognizer.train(faces, np.array(ids))
recognizer.write('trainer.yml')
