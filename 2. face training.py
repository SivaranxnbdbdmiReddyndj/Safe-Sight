import cv2
import numpy as np
from PIL import Image
import os

# Path to the dataset folder containing subfolders for each user
dataset_path = r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\dataset"

# Create an LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade classifier for face detection
cascade_path = r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

# Function to get images and labels for training from all subfolders
def get_images_and_labels(path):
    face_samples = []
    ids = []

    # Iterate through all subfolders in the dataset directory
    for user_folder in os.listdir(path):
        user_path = os.path.join(path, user_folder)
        if not os.path.isdir(user_path):
            continue  # Skip if not a directory

        # Iterate through all images in the user's subfolder
        for image_file in os.listdir(user_path):
            image_path = os.path.join(user_path, image_file)

            try:
                # Open the image and convert to grayscale
                img = Image.open(image_path).convert('L')
                img_np = np.array(img, 'uint8')

                # Extract the user ID from the filename
                face_id = int(os.path.split(image_file)[-1].split(".")[1])

                # Detect faces in the image
                faces = faceCascade.detectMultiScale(
                    img_np,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Add each detected face and its corresponding ID to the lists
                for (x, y, w, h) in faces:
                    face_samples.append(img_np[y:y + h, x:x + w])
                    ids.append(face_id)
            except Exception as e:
                print(f"[ERROR] Skipping {image_path}: {str(e)}")

    return face_samples, ids

print("[INFO] Training faces. It will take a few seconds. Please wait...")

# Get the face samples and their corresponding IDs
faces, ids = get_images_and_labels(dataset_path)

if len(faces) == 0:
    print("[ERROR] No faces detected in the dataset. Make sure the images are correct and try again.")
    exit()

# Train the recognizer with the collected face samples and IDs
recognizer.train(faces, np.array(ids))

# Save the trained model
model_path = r"C:\Users\sivar\Downloads\Smart-home-using-face-recognixation-main\Smart-home-using-face-recognixation-main\trainer\trainer.yml"
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))

recognizer.write(model_path)
print(f"[INFO] Model trained and saved at {model_path}")
