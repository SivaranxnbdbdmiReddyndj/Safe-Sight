import numpy as np
import cv2

# Load the Haar Cascade classifiers
faceCascade = cv2.CascadeClassifier(r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\FaceDetection\Cascades\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\FaceDetection\Cascades\haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier(r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\FaceDetection\Cascades\haarcascade_smile.xml")

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

print("Press 'ESC' to exit.")

while True:
    # Capture frame from the camera
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally and convert it to grayscale
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(25, 25)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles within the face
        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=15,
            minSize=(25, 25)
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Face, Eye, and Smile Detection', img)

    # Exit when 'ESC' is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
