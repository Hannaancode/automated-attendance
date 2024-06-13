Automated Attendance
Automated Attendance is a project designed to streamline the process of tracking attendance using AI technology. It offers different functionalities for processing attendance data based on different inputs.

Usage
For Oak camera processing, run main.py.
For video file processing and live camera processing using an RTSP link, run radio.py.
For registering faces in the database for the Oak camera, use register.py.
For registering faces in the database for live camera processing and video processing, use newradio.py.
Installation
Make sure to install the necessary modules and libraries before running the project.

Some concepts used :

Neural Networks:

Face Detection Neural Network: Utilizes a pre-trained MobileNet for detecting faces in the video stream.
Head Pose Estimation Neural Network: Uses a pre-trained model to estimate the head pose from detected faces.
Face Recognition Neural Network: Employs an ArcFace model for recognizing and verifying faces based on facial features.
Emotion Recognition Neural Network: Uses a pre-trained model to identify emotions from detected faces.
Deep Learning Models:

MobileNet: A lightweight deep learning model designed for efficient image classification and detection tasks on mobile and edge devices.
ArcFace: A deep learning model that uses an additive angular margin loss to achieve high performance in face recognition tasks.
Head Pose Estimation Model: Typically a CNN-based model that estimates the pitch, yaw, and roll angles of the head.
Emotion Recognition Model: A neural network trained to classify facial expressions into different emotion categories.
Computer Vision:

Image Manipulation: The code uses OpenCV and DepthAI's ImageManip node to preprocess and manipulate images before feeding them into neural networks.
Bounding Box Calculation: The frame_norm function normalizes and calculates bounding boxes for detected faces.
Text Rendering: Uses OpenCV to render text annotations (such as detected emotions and names) on the video frames.
Real-time Processing:

Synchronization: The TwoStageHostSeqSync class is used to synchronize messages from different neural networks (e.g., face detections, recognitions, and emotions) to ensure that the data processed is from the same frame.
Pipeline Creation: The DepthAI pipeline is created to define the flow of data from the camera, through various neural networks, and back to the host application.
Data Management:

Database Management: The FaceRecognition class reads and writes face embeddings to a local database, allowing the system to recognize previously seen faces.
Data Saving and Loading: The code saves detected face information and sends data to a server via HTTP requests.
Argument Parsing:

The argparse library is used to parse command-line arguments for configuring the program, such as specifying the name for database saving.
Utility Functions and Classes:

TextHelper: A utility class to simplify text rendering on video frames.
Cosine Distance Calculation: Used for comparing face embeddings to determine identity matches.

Inference Engine (OpenVINO):

IECore: Initializes the Inference Engine Core for executing neural networks on different hardware accelerators.
Face Detection and Recognition:

Face Detection Model: Uses a pre-trained model to detect faces in images.
Face Recognition Model: Uses a pre-trained model to recognize faces by comparing embeddings.
Emotion Recognition:

Emotion Recognition Model: Uses a pre-trained model to classify facial expressions into different emotions.
Head Pose Estimation:

Head Pose Model: Uses a pre-trained model to estimate head pose angles (yaw, pitch).
Cosine Similarity:

cosine_similarity: Calculates the similarity between face embeddings using cosine similarity.
File Handling and Data Storage:

Pickle: Loads and saves face embeddings from and to files.
OpenCV (cv2): Handles video capture, image processing, and displaying frames.
JSON: Formats and writes face data into a JSON structure for further processing or storage.
GUI Development:

Tkinter: Creates graphical user interfaces for video file selection and camera ID input.
Tkinter ttk: Provides themed widgets for the Tkinter GUI.
Networking and HTTP Requests:

Requests Library: Sends HTTP POST requests to a server with detected face data.
Real-time Video Processing:

RTSP Stream: Processes live video feeds from RTSP cameras.
Threading:

Threading Library: Manages concurrent execution for potentially handling GUI updates or video processing.
Date and Time Handling:

Datetime Library: Manages timestamps for video frames and calculates video creation date.
File and Directory Operations:

OS Library: Handles file paths, directories, and file existence checks.
Error Handling:

Try-Except Blocks: Manages potential errors, especially for file operations and video capture.


Main.py working summary:

Pipeline Setup:
The code sets up a DepthAI pipeline, configuring various nodes for color camera input, image manipulation, and neural network processing.
Real-time Processing:
Frames from the camera are processed in real-time, with detections from the face detection neural network used to crop and resize regions of interest for further analysis.
Face and Emotion Recognition:
Cropped face images are fed into the face recognition and emotion recognition neural networks.
Results are synchronized and processed to annotate the video frames with detected emotions and recognized identities.
Data Management:
Detected faces are stored in a local database, and new unknown faces are saved with unique identifiers.
Detected face information is periodically saved to a file and sent to a server.

Contributors
Abdul Hannan
