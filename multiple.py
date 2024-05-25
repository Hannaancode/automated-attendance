import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import pickle
from openvino.inference_engine import IECore
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import datetime
import ffmpeg
import requests
import threading
import time
from tkinter import ttk

# Initialize the Inference Engine Core
ie = IECore()

# Load the face detection model
face_detection_net = ie.read_network(model='face-detection-retail-0004.xml',
                                     weights='face-detection-retail-0004.bin')
face_detection_exec_net = ie.load_network(network=face_detection_net, device_name='CPU')

# Load the face recognition model
face_recognition_net = ie.read_network(model='face-recognition-arcface-112x112.xml',
                                        weights='face-recognition-arcface-112x112.bin')
face_recognition_exec_net = ie.load_network(network=face_recognition_net, device_name='CPU')

# Load the emotion recognition model
emotion_recognition_net = ie.read_network(model='emotions-recognition-retail-0003.xml',
                                          weights='emotions-recognition-retail-0003.bin')
emotion_recognition_exec_net = ie.load_network(network=emotion_recognition_net, device_name='CPU')

head_pose_net = ie.read_network(model='head-pose-estimation-adas-0001.xml',
                                 weights='head-pose-estimation-adas-0001.bin')
head_pose_exec_net = ie.load_network(network=head_pose_net, device_name='CPU')



# Path to the directory where the face database is saved
database_dir = 'face_database'

# Path to the directory where the video data will be saved
video_data_dir = 'video_data'
os.makedirs(video_data_dir, exist_ok=True)

# Load the embeddings from the database
face_database = {}
for file_name in os.listdir(database_dir):
    if file_name.endswith('.pkl'):
        name = os.path.splitext(file_name)[0]
        with open(os.path.join(database_dir, file_name), 'rb') as f:
            embeddings = pickle.load(f)
            face_database[name] = embeddings

# Function to select the video file
def select_video_file():
    global video_file
    video_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Video File",
                                            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    video_dir_entry.delete(0, tk.END)
    video_dir_entry.insert(0, video_file)

# Function to get the creation date of a video file
def get_video_creation_date(video_path):
    try:
        creation_time = os.path.getctime(video_path)
        creation_date = datetime.datetime.fromtimestamp(creation_time)
        return creation_date
    except Exception as e:
        print(f"Error: {e}")
        return None


def save_unknown_face(face_img, unknown_faces_dir):
    # Create the directory if it doesn't exist
    os.makedirs(unknown_faces_dir, exist_ok=True)

    # Generate a unique filename for the image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(unknown_faces_dir, f"unknown_face_{timestamp}.jpg")

    # Save the face image
    cv2.imwrite(filename, face_img)
    

# Define the function to process a single video stream
def process_video_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error opening video stream: {rtsp_url}")
        return

    # Load the face detection and recognition models
    # Assuming face_detection_exec_net and face_recognition_exec_net are already defined

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face detection
        resized_frame = cv2.resize(frame, (300, 300))
        input_frame = np.expand_dims(resized_frame, axis=0)
        input_frame_transposed = np.transpose(input_frame, (0, 3, 1, 2))
        face_detection_res = face_detection_exec_net.infer(inputs={'data': input_frame_transposed})
        detections = face_detection_res['detection_out']

        # Loop over the detected faces
        for detection in detections[0][0]:
            if detection[2] > 0.5:  # Confidence threshold
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])

                face_img = frame[ymin:ymax, xmin:xmax]

                if face_img.size != 0:  # Check if face_img is not empty
                    resized_face_img = cv2.resize(face_img, (112, 112))  # Resize to 112x112

                    # Perform face recognition
                    face_recognition_res = face_recognition_exec_net.infer(inputs={'data': resized_face_img.transpose((2, 0, 1)).reshape(1, 3, 112, 112)})
                    embeddings = face_recognition_res['pre_fc1/Fused_Add_'][0]
                    embeddings /= np.linalg.norm(embeddings)  # Normalize embeddings

                    # Compare with embeddings of known faces
                    best_match_name = None
                    best_match_score = -1
                    for name, db_embeddings in face_database.items():
                        similarity = cosine_similarity([embeddings], db_embeddings)[0][0]  # Extract the single similarity value
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_name = name

                    if best_match_score > 0.35:  # Threshold for recognition
                        # Perform emotion recognition
                        # Assuming emotion_recognition_exec_net is already defined
                        resized_face_img = cv2.resize(face_img, (64, 64))
                        padded_face_img = np.pad(resized_face_img, ((0, 64 - resized_face_img.shape[0]),
                                                                     (0, 64 - resized_face_img.shape[1]),
                                                                     (0, 0)), mode='constant')
                        padded_face_img = padded_face_img.transpose((2, 0, 1))
                        padded_face_img = np.expand_dims(padded_face_img, axis=0)
                        emotion_recognition_res = emotion_recognition_exec_net.infer(inputs={'data': padded_face_img})

                        cleaned_name = re.sub(r'[^a-zA-Z]', '', best_match_name)

                        # Extract only numeric characters for the ID
                        id_number = re.sub(r'[^0-9]', '', best_match_name)

                        # Use 'cleaned_name' for display and 'id_number' as the ID
                        emotion_label = np.argmax(emotion_recognition_res['prob_emotion'][0])
                        emotion_text = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger'][emotion_label]

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 225), 2)  # Draw a rectangle around the face
                        cv2.putText(frame, f'{cleaned_name} ({id_number}) - {emotion_text}', (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display name and emotion

                        # Display face detection boxes with details in the video
                        cv2.imshow(rtsp_url, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# List of RTSP URLs for each video stream
rtsp_urls = [
    "rtsp://admin:tvmCX123@192.168.29.194:554/Streaming/Channels/101/",
    "rtsp://admin:tvmCX123@192.168.29.194:554/Streaming/Channels/201/"
]

# Create and start a thread for each video stream
threads = []
for rtsp_url in rtsp_urls:
    thread = threading.Thread(target=process_video_stream, args=(rtsp_url,))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()
