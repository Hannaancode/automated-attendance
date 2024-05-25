import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize the FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize the MTCNN for face detection
mtcnn = MTCNN()

# Path to the directory where the face database is saved
face_database_path = 'face_database.npy'

# Load the face database if it exists, or create a new empty database
if os.path.exists(face_database_path):
    face_database = np.load(face_database_path, allow_pickle=True).item()
else:
    face_database = {}

# Ask for the person's name
name = input("Enter the person's name: ")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame from webcam")
        break

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Wait for the user to press 'c' to capture the face
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None and len(boxes) == 1:
            x, y, w, h = [int(val) for val in boxes[0]]
            face_img = frame[y:y+h, x:x+w]
            aligned_face = mtcnn.align(face_img)

            # Display the face being stored
            cv2.imshow('Face', aligned_face)
            cv2.waitKey(2000)  # Display the face for 2 seconds

            # Convert the face image to a tensor
            aligned_face_tensor = torch.tensor(aligned_face.transpose(2, 0, 1), dtype=torch.float32)

            # Normalize the face image
            aligned_face_tensor /= 255.0

            # Extract face embeddings using FaceNet
            embeddings = facenet_model(aligned_face_tensor.unsqueeze(0)).detach().numpy()

            # Store the face embeddings in the database
            face_database[name] = embeddings.flatten()

            # Save the updated face database
            np.save(face_database_path, face_database)

            print(f"Face captured and saved for {name}")
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
