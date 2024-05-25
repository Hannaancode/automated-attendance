# coding=utf-8
import os
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync1 import TwoStageHostSeqSync
import tkinter as tk
from tkinter import messagebox
import argparse
import sys

import os

import os
import sys
import tkinter as tk
from tkinter import messagebox

class RegistrationInterface:
    def __init__(self,  name_to_id_mapping, parser):
        self.parser = parser  # Store the parser
        self.main_py_path = "main.py"  # Path to the main.py file

        self.root = tk.Tk()
        self.root.title("Registration Interface")

        self.name_label = tk.Label(self.root, text="name:")
        self.name_label.pack()

        self.name_entry = tk.Entry(self.root)
        self.name_entry.pack()

        self.id_label = tk.Label(self.root, text="Custom ID:")
        self.id_label.pack()

        self.id_entry = tk.Entry(self.root)
        self.id_entry.pack()

        self.register_button = tk.Button(self.root, text="Register", command=self.register)
        self.register_button.pack()

        # Add new attributes to store the entered name and ID
        self.entered_name = None
        self.entered_id = None

        # Add an attribute to track if registration is successful
        self.registration_successful = False

        # Bind the close event of the GUI window to the on_window_close function
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def register(self):
        # Get the entered name and custom ID from the entry fields
        name = self.name_entry.get()
        custom_id = self.id_entry.get()

        # Check if both name and ID are entered
        if not name or not custom_id:
            # Display an error message
            messagebox.showerror("Error", "Please enter both name and ID.")
            return  # Exit the method without proceeding further

        # Write the entered name and ID to the main.py file
        self.write_to_main_py(name, custom_id)

        # Set the entered name and custom ID as attributes
        self.entered_name = name
        self.entered_id = custom_id
        self.registration_successful = True

        # Close the registration window
        self.root.destroy()


    

    def on_window_close(self):
        # Close the GUI window and terminate the program when the close button is pressed
        self.root.destroy()
        sys.exit()

    def run(self):
        self.root.mainloop()

        # Method to write the entered name and ID to the main.py file
    def write_to_main_py(self, name, custom_id):
    # Check if the main.py file exists
     if not os.path.exists(self.main_py_path):
         messagebox.showerror("Error", f"{self.main_py_path} does not exist.")
         return

    # Open the main.py file and add the entered name and ID to the name_to_id_mapping dictionary
     with open(self.main_py_path, "r+") as main_py_file:
         lines = main_py_file.readlines()
         main_py_file.seek(0)

         # Flag to determine if name_to_id_mapping has been found
         mapping_found = False
         mapping_closed = False
         name_written = False  # Flag to track if the name and ID have been written

         for line in lines:
             if "name_to_id_mapping" in line and not mapping_found:
                 mapping_found = True
                 main_py_file.write(line)  # Write the line with the dictionary start
                 continue

             if mapping_found and not mapping_closed:
                 # Check if the dictionary is closed
                 if line.strip() == "}":
                     mapping_closed = True

                 # Write the entered name and ID inside the dictionary if not already written
                 if not name_written:
                     main_py_file.write(f'    "{name}": {custom_id},\n')
                     name_written = True  # Set the flag to indicate that name and ID have been written

             main_py_file.write(line)

         # If name_to_id_mapping was not found, write it at the end of the file
         if not mapping_found:
             main_py_file.write(f"name_to_id_mapping = {{\n    \"{name}\": {custom_id},\n}}\n")





def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

VIDEO_SIZE = (1072, 1072)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)

class FaceRecognition:
    def __init__(self, db_path, name, assigned_id=None) -> None:
        self.read_db(db_path)
        self.name = name
        self.assigned_id = assigned_id
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

        # If assigned_id is not provided, use an empty dictionary
        if self.assigned_id is None:
            self.assigned_id = {}

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        if self.name is None or self.assigned_id is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name or --id wasn't specified")
                self.printed = True
            return
        existing_db_path = f"{databases}/{self.name}.npz"
        existing_db_size = os.path.getsize(existing_db_path) if os.path.exists(existing_db_path) else 0

        # Set the maximum allowed size for the face database (e.g., 100 KB)
        max_allowed_size = 200* 1024

        if existing_db_size >= max_allowed_size:
            print(f"Face database for {self.name} has reached the maximum allowed size. Exiting...")
            os._exit(1)
        print('Saving face...')
        try:
            with np.load(f"{databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{databases}/{self.name}", *db_)
        self.adding_new = False

    def get_id_by_name(self, name):
        # Retrieve assigned ID by name
        return self.assigned_id.get(name)

# Define a mapping of names to specific IDs
name_to_id_mapping = {
    "PersonC": 3,
    # Add more names and IDs as needed
}

# Create the argument parser

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help=f"Name of the person for database saving )")



# Create the registration interface
registration_interface = RegistrationInterface(name_to_id_mapping, parser)

# Initialize args outside the loop
args = parser.parse_args([])

# Run the Tkinter main loop to display the registration interface
registration_interface.run()

# Access the entered name and custom ID after the Tkinter main loop
entered_name = registration_interface.entered_name
entered_id = registration_interface.entered_id

# Update args.name with the entered name
args.name = entered_name
registration_interface.write_to_main_py(entered_name, entered_id)

print(f"Entered Name: {entered_name}, Entered Custom ID: {entered_id}")

pipeline = dai.Pipeline()

# Remaining pipeline setup code...
# Define remaining pipeline components
print("Creating Color Camera...")
cam = pipeline.create(dai.node.ColorCamera)
# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(1072, 1072)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

host_face_out = pipeline.create(dai.node.XLinkOut)
host_face_out.setStreamName('color')
cam.video.link(host_face_out.input)

# ImageManip as a workaround to have more frames in the pool.
# cam.preview can only have 4 frames in the pool before it will
# wait (freeze). Copying frames and setting ImageManip pool size to
# higher number will fix this issue.
copy_manip = pipeline.create(dai.node.ImageManip)
cam.preview.link(copy_manip.inputImage)
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(1072*1072*3)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
copy_manip.out.link(face_det_manip.inputImage)

# NeuralNetwork
print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
# Link Face ImageManip -> Face detection NN node
face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
# We also interested in sequence number for syncing
face_det_nn.passthrough.link(script.inputs['face_pass'])

copy_manip.out.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

print("Creating Head pose estimation NN")

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.setWaitForConfigInput(True)
script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
script.outputs['manip_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

print("Creating face recognition ImageManip/NN")

face_rec_manip = pipeline.create(dai.node.ImageManip)
face_rec_manip.initialConfig.setResize(112, 112)
face_rec_manip.inputConfig.setWaitForMessage(True)

script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
script.outputs['manip2_img'].link(face_rec_manip.inputImage)

face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
face_rec_manip.out.link(face_rec_nn.input)

arc_xout = pipeline.create(dai.node.XLinkOut)
arc_xout.setStreamName('recognition')
face_rec_nn.out.link(arc_xout.input)

with dai.Device(pipeline) as device:
    facerec = FaceRecognition(databases, args.name, name_to_id_mapping)
    sync = TwoStageHostSeqSync()
    text = TextHelper()

    queues = {}
    # Create output queues
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and face recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            dets = msgs["detection"].detections

            for i, detection in enumerate(dets):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

                features = np.array(msgs["recognition"][i].getFirstLayerFp16())
                conf, name = facerec.new_recognition(features)

                # Get assigned ID by name
                id = facerec.get_id_by_name(name)

                text.putText(frame, f"{name}: ID _{id}_ {(100*conf):.0f}%", (bbox[0] + 10, bbox[1] + 35))

            cv2.imshow("color", cv2.resize(frame, (800, 800)))

        if cv2.waitKey(1) == ord('q'):
            break
