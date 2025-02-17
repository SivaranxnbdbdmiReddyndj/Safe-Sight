import cv2
import numpy as np
import os
import pyttsx3  # Text-to-speech library

# Paths to the trained model and Haar Cascade
model_path = r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\trainer\trainer.yml"
cascade_path = r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\haarcascade_frontalface_default.xml"

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Ensure the cascade and model files exist
if not os.path.isfile(cascade_path):
    print(f"[ERROR] Haar Cascade file '{cascade_path}' does not exist.")
    exit()
if not os.path.isfile(model_path):
    print(f"[ERROR] Trained model file '{model_path}' does not exist.")
    exit()

# Load the recognizer and Haar Cascade classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Font for displaying text on the screen
font = cv2.FONT_HERSHEY_SIMPLEX

# List of names corresponding to IDs (Ensure this matches the IDs in the dataset)
names = ['Unknown', 'Arun',]  # Add names as required

# Initialize the webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Unable to access the webcam.")
    exit()

cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

# Define the minimum window size for face detection
minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

print("[INFO] Starting real-time face recognition. Press 'ESC' to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image from webcam.")
        break

    img = cv2.flip(img, 1)  # Flip the image horizontally
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,  # Adjusted for finer detection
        minNeighbors=5,   # Increased for stricter detection
        minSize=(minW, minH)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Determine the name and confidence
        if confidence < 50:  # Threshold adjusted for accuracy
            name = names[id] if id < len(names) else "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"

        # Display name and confidence
        cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Announce the name using TTS
        if name != "Unknown":
            engine.say(f"Hello, {name}")
            engine.runAndWait()

    # Display the video feed with recognition
    cv2.imshow('Face Recognition', img)

    # Exit loop if 'ESC' is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Cleanup resources
print("[INFO] Exiting the program and cleaning up.")
cam.release()
cv2.destroyAllWindows()
