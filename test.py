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
import subprocess

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


def align_face(image, landmarks):
    # Calculate the angle of rotation based on the eye landmarks
    eyes_center = ((landmarks[0][0] + landmarks[1][0]) // 2, (landmarks[0][1] + landmarks[1][1]) // 2)
    dy = landmarks[1][1] - landmarks[0][1]
    dx = landmarks[1][0] - landmarks[0][0]
    angle = np.degrees(np.arctan2(dy, dx)) - 90  # -90 to make the eyes horizontal

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_face

def enhance_image(image, landmarks):
    # Align the face
    aligned_face = align_face(image, landmarks)

    # Apply image enhancement techniques here
    # For example, sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced_image = cv2.filter2D(aligned_face, -1, kernel)
    
    return enhanced_image            

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
    


def estimate_head_pose(face_img):
    resized_face_img = cv2.resize(face_img, (60, 60))  # Resize to 60x60 for head pose estimation
    head_pose_input = np.transpose(resized_face_img, (2, 0, 1))  # Change the order to (C, H, W)
    head_pose_input = np.expand_dims(head_pose_input, axis=0)
    head_pose_res = head_pose_exec_net.infer(inputs={'data': head_pose_input})
    yaw_angle = int(head_pose_res['angle_y_fc'][0][0])  # Yaw angle (left-right)
    pitch_angle = int(head_pose_res['angle_p_fc'][0][0])  # Pitch angle (up-down)

    # Calculate adjustments for xmin, ymin, xmax, ymax based on yaw and pitch angles
    x_adjustment = yaw_angle
    y_adjustment = pitch_angle

    return x_adjustment, y_adjustment



def start_processing():
    global video_file
    video_file = video_dir_entry.get()
    cap = cv2.VideoCapture(video_file)
    camera_id = camera_id_entry.get()

    # Get the creation date of the video file
    creation_date = get_video_creation_date(video_file)
    if creation_date is None:
        creation_date = 0
        return

    # Initialize variables to store detected faces and their information
    detected_faces = {}
    processed_faces = set()

    unknown_faces_dir = os.path.join(database_dir, 'unknown_faces')

    # Write the detected faces information to a single text file
    with open(os.path.join(video_data_dir, 'video_data.txt'), 'w') as txt_file:
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

                        if best_match_score > 0.45:  # Threshold for recognition
                            # Perform emotion recognition
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

                            # Estimate head pose
                            head_pose_angles = estimate_head_pose(face_img)
                            

                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 225), 2)  # Draw a rectangle around the face
                            cv2.putText(frame, f'{cleaned_name} ({id_number}) - {emotion_text}', (xmin, ymin - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display name and emotion

                            # Calculate the timestamp based on the creation date of the video file
                            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds
                            timestamp = (current_time - creation_date.timestamp()) + 1

                            face_id = hash(best_match_name)
                            if face_id not in processed_faces:
                                face_data = {
                                    "name": cleaned_name,
                                    "id": id_number,
                                    "emotion": emotion_text,
                                    "timestamp": (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) + (creation_date.timestamp()),
                                    "camera ID":camera_id
                                }
                                data = {
                                    "id": int(id_number),
                                    "emotion": emotion_text,
                                    "timestamp": str((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) + (creation_date.timestamp())),
                                    "camera ID":camera_id
                                    }
                                url = 'http://18.188.42.141:8080/face/detection/insert'
                                with open('token.txt', 'r') as file:
                                     token = file.read().strip()
                                     print(token)

                                headers = {
                                      'Authorization': 'Bearer '+token,
                                      'Content-Type': 'application/json'
                                      }
                                print(data)
                                #response = requests.post(url, headers=headers, json=data)
                                #if response.text:  # Check if response is not empty
                                      #print(response.json())
                               # else:
                                    #print("Empty response received from the server.")

                                

                                txt_file.write(json.dumps(face_data) + '\n')
                                processed_faces.add(face_id)
                        else:  # Face not recognized
                            cleaned_name = 'Unknown'
                            id_number = 'Unknown'
                            emotion_text = 'Unknown'
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 225), 2)  # Draw a rectangle around the face
                            cv2.putText(frame, f'{cleaned_name} ({id_number}) - {emotion_text}', (xmin, ymin - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Display name and emotion
                            save_unknown_face(face_img, unknown_faces_dir)

            # Display the frame
            cv2.imshow('Video', frame)

            # Increase the frame rate for faster playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    
def start_live_processing():
    # Create GUI for entering camera ID
    def on_submit():
        nonlocal camera_id_entry
        camera_id = camera_id_entry.get()
        cap = cv2.VideoCapture(f"rtsp://admin:tvmCX123@192.168.29.194:554/Streaming/Channels/{camera_id}/")
        if camera_id == '101':
            location = "Frontend"
        elif camera_id == '201':
            location =  "Backend"
        elif camera_id == '301':
            location =  "Frontend"
        elif camera_id == '401':
            location =  "Backend"    
        else:
            location = "None"
            return

            
        cap = cv2.VideoCapture(f"rtsp://admin:tvmCX123@192.168.29.194:554/Streaming/Channels/{camera_id}/")    
        

        # Initialize variables to store detected faces and their information
        detected_faces = {}
        processed_faces = set()
        unknown_faces_dir = os.path.join(database_dir, 'unknown_faces')

        # Write the detected faces information to a single text file
        with open(os.path.join(video_data_dir, 'live_video_data.txt'), 'a') as txt_file:
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

                            if best_match_score > 0.45:  # Threshold for recognition
                                # Perform emotion recognition
                                aligned_face_img = align_face(face_img, landmarks)  
                                enhanced_face_img = enhance_image(aligned_face_img)
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

                                # Estimate head pose
                                head_pose_angles = estimate_head_pose(face_img)

                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 225), 2)  # Draw a rectangle around the face
                                cv2.putText(frame, f'{cleaned_name} ({id_number}) - {emotion_text}', (xmin, ymin - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display name and emotion

                                # Display face detection boxes with details in the video
                                cv2.imshow('Live Video', frame)

                                # Calculate the timestamp based on the current time
                                current_time = datetime.datetime.now()
                                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

                                face_id = hash(best_match_name)
                                if face_id not in processed_faces:
                                    face_data = {
                                        "name": cleaned_name,
                                        "id": id_number,
                                        "emotion": emotion_text,
                                        "timestamp": timestamp
                                    }
                                    data = {
                                        "id": int(id_number),
                                        "emotion": emotion_text,
                                        "timestamp": timestamp,
                                        "camera ID": camera_id,
                                        "location": location
                                    }
                                    url = 'http://18.188.42.141:8080/face/detection/insert'
                                    with open('token.txt', 'r') as file:
                                        token = file.read().strip()
                                        print(token)

                                    headers = {
                                        'Authorization': 'Bearer ' + token,
                                        'Content-Type': 'application/json'
                                    }
                                    print(data)

                                    #response = requests.post(url, headers=headers, json=data)
                                    #if response.text:  # Check if response is not empty
                                        #print(response.json())
                                    #else:
                                        #print("Empty response received from the server.")

                                    txt_file.write(json.dumps(data) + '\n')
                                    processed_faces.add(face_id)
                            else:
                                cleaned_name = 'Unknown'
                                id_number = 'Unknown'
                                emotion_text = 'Unknown'
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 225), 2)  # Draw a rectangle around the face
                                cv2.putText(frame, f'{cleaned_name} ({id_number}) - {emotion_text}', (xmin, ymin - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Display name and emotion
                                save_unknown_face(face_img, unknown_faces_dir)

                # Display the frame
                cv2.imshow('Live Video', frame)

                # Exit the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    root = tk.Tk()
    root.title("Camera ID Input")

    # Create a label and an entry for entering the camera ID
    camera_id_label = tk.Label(root, text="Enter Camera ID:")
    camera_id_label.pack()

    camera_id_entry = tk.Entry(root)
    camera_id_entry.pack()

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack()

    root.mainloop()





# Create the main GUI window
root = tk.Tk()
root.title("Video Directory Selector")


# Label and entry for video directory selection
video_dir_label = tk.Label(root, text="Video Directory:")
video_dir_label.pack()
video_dir_entry = tk.Entry(root, width=50)
video_dir_entry.pack()

# Label and entry for camera ID input
camera_id_label = tk.Label(root, text="Camera ID:")
camera_id_label.pack()
camera_id_entry = tk.Entry(root, width=50)
camera_id_entry.pack()

# Button to select video directory
select_button = tk.Button(root, text="Select Video File", command=select_video_file)
select_button.pack()

# Button to start processing
process_button = tk.Button(root, text="Start Processing", command=start_processing)
process_button.pack()

# Button to start live video processing
process_button = tk.Button(root, text="Start Live Processing", command=start_live_processing)
process_button.pack()

root.mainloop()
