import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize_face():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 50:
                cap.release()
                cv2.destroyAllWindows()
                return True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return False

@app.route('/unlock', methods=['POST'])
def unlock():
    if recognize_face():
        return jsonify({"status": "unlocked"})
    else:
        return jsonify({"status": "locked"})

if __name__ == "__main__":
    app.run(debug=True)
