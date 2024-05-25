import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize the FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize the MTCNN for face detection
mtcnn = MTCNN()

# Load the face database
face_database = np.load('face_database.npy', allow_pickle=True).item()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame")
        break

    # Perform face detection using MTCNN
    boxes, probs = mtcnn.detect(frame)

    if boxes is not None:
        for i, box in enumerate(boxes):
            # Crop and align the face
            x, y, w, h = [int(val) for val in box]
            face_img = frame[y:y+h, x:x+w]
            aligned_face = mtcnn.align(face_img)

            # Convert the face image to a tensor
            aligned_face_tensor = torch.tensor(aligned_face.transpose(2, 0, 1), dtype=torch.float32)

            # Normalize the face image
            aligned_face_tensor /= 255.0

            # Extract face embeddings using FaceNet
            embeddings = facenet_model(aligned_face_tensor.unsqueeze(0)).detach().numpy()

            # Compare with embeddings of known faces
            best_match_name = 'Unknown'
            best_match_score = -1
            for name, db_embeddings in face_database.items():
                similarity = np.dot(embeddings.flatten(), db_embeddings.flatten())
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_name = name

            # Display the face and recognition result
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, best_match_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
