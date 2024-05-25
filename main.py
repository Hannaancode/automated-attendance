import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync
import json
import time
from datetime import datetime
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()


emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger'] 


def pad_emotions(emotions, length):
    return emotions + ['Unknown'] * (length - len(emotions))

VIDEO_SIZE = (1072, 1072)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)


def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)



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
        self.last_stored_times = {}
        self.unknown_counter = 1
        self.databases_path = "./databases"

        # If assigned_id is not provided, use an empty dictionary
        if self.assigned_id is None:
            self.assigned_id = {}
        self.detected_faces_info = []    

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("arrays {} and {} have different shapes: {} and {}".format(a, b, a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                # Repeat the elements of j to match the length of results
                b_reshaped = np.tile(j, len(results) // len(j) + 1)[:len(results)]
                conf_ = self.cosine_distance(b_reshaped, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")

        # Check if the detected face is unknown
        if name[1] == "UNKNOWN":


            # Save each unknown face separately
            self.create_db(results)

        return name



    def read_db(self, databases):
        self.labels = []
        for file in os.listdir(databases):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    
    def create_db(self, results):
        # Check the number of unknown faces present on the screen
        num_unknown_faces_detected = sum(1 for n in self.detected_faces_info if n["name"] == "UNKNOWN")

        # Save unknown faces, ensuring each one is within 30 KB limit
        for i in range(num_unknown_faces_detected):
            new_name = f"UNKNOWN_{self.unknown_counter}_{i + 1}"
            np.savez_compressed(f"{databases}/{new_name}", np.array(results))
            file_size = os.path.getsize(f"{databases}/{new_name}.npz")
            if file_size >= 30 * 1024:  # 30 KB in bytes
                self.unknown_counter += 1
                break  # Stop saving more unknown faces if the limit is reached
            self.read_db(databases)
            self.last_stored_times[new_name] = time.time()

        # Proceed with the original logic to save the face if no unknown face was saved in the current iteration
        if num_unknown_faces_detected == 0:
            name = f"UNKNOWN_{self.unknown_counter}"
            print('Saving face...')
            try:
                with np.load(f"{databases}/{name}.npz") as db:
                    db_ = [db[j] for j in db.files][:]
            except FileNotFoundError:
                db_ = []

            db_.append(np.array(results))
            np.savez_compressed(f"{databases}/{name}", *db_)

            file_size = os.path.getsize(f"{databases}/{name}.npz")
            if file_size >= 30 * 1024:  # 30 KB in bytes
                self.unknown_counter += 1
                self.read_db(databases)








        


        
    

    def save_detected_faces_info(self, file_path="./data/detected_faces_info.txt"):
        # Get the current time
        current_time = datetime.now().timestamp()

        # Iterate over detected faces info
        for info in self.detected_faces_info:
            name = info["name"]
            confidence = info["confidence"]
            
            # Save only if confidence level is above 80%
            if confidence > 0.75:
                info["timestamp"] = current_time
                
                # Check if the face name has been stored before
                if name not in self.last_stored_times:
                    self.last_stored_times[name] = 0
                
                # Check if 50 seconds have passed since the last storage for this face name
                if current_time - self.last_stored_times[name] >= 50:
                    # Open the file in append mode
                    with open(file_path, "a") as file:
                        # Write the detected face info to the file
                        file.write(json.dumps(info) + "\n")

                    # Construct the 'data' dictionary
                    data = {
                        "id": info["id"],
                        "emotion": info["emotion"],
                        "timestamp": current_time,
                        "camera ID": "oak camera"
                    }

                    # Send the data to the server
                    url = 'http://18.188.42.141:8080/face/detection/insert'
                    with open('token.txt', 'r') as file:
                        token = file.read().strip()
                        print(token)

                    headers = {
                        'Authorization': 'Bearer ' + token,
                        'Content-Type': 'application/json'
                    }

                    # Print the 'data' dictionary
                    print(data)

                    # Send the data to the server
                    # response = requests.post(url, headers=headers, json=data)
                    # if response.text:  # Check if response is not empty
                    #     print(response.json())
                    # else:
                    #     print("Empty response received from the server.")

                    # Update the last stored time for this face name
                    self.last_stored_times[name] = current_time

        # Clear the detected_faces_info list after processing
        self.detected_faces_info.clear()

        
    def clear_detected_faces_info(self):
        self.detected_faces_info = []
        
    def get_id_by_name(self, name):
        # Retrieve assigned ID by name
        return self.assigned_id.get(name)
        
    def get_id_by_name(self, name):
        # Retrieve assigned ID by name
        return self.assigned_id.get(name)

name_to_id_mapping = {
    "ggg898": 222,
    "ggg898": 222,
    "hannaan": 111,
    "hannaan": 111,
                
    456: "jack",
    333: "jack",
    
   
                    
    "ashiq": 45,
    "ashiq": 45,
   
    
                }   

print("Creating pipeline...")
pipeline = dai.Pipeline()

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
headpose_manip.inputConfig.setWaitForMessage(True)
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

# This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
# face that was detected by the face-detection NN.

image_manip_script = pipeline.create(dai.node.Script)
face_det_nn.out.link(image_manip_script.inputs['face_det_in'])

# Only send metadata, we are only interested in timestamp, so we can sync
# depth frames with NN output
face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])
copy_manip.out.link(image_manip_script.inputs['preview'])

image_manip_script.setScript("""
import time
msgs = dict()

def add_msg(msg, name, seq = None):
    global msgs
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    # node.warn(f"New msg {name}, seq {seq}")

    # Each seq number has it's own dict of msgs
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

    # To avoid freezing (not necessary for this ObjDet model)
    if 15 < len(msgs):
        node.warn(f"Removing first element! len {len(msgs)}")
        msgs.popitem() # Remove first element

def get_msgs():
    global msgs
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
        # node.warn(f"Checking sync {seq}")

        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 2: # 1 frame, 1 detection
            for rm in seq_remove:
                del msgs[rm]
            # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
            return syncMsgs # Returned synced msgs
    return None

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb

while True:
    time.sleep(0.001) # Avoid lazy looping

    preview = node.io['preview'].tryGet()
    if preview is not None:
        add_msg(preview, 'preview')

    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
        passthrough = node.io['passthrough'].get()
        seq = passthrough.getSequenceNum()
        add_msg(face_dets, 'dets', seq)

    sync_msgs = get_msgs()
    if sync_msgs is not None:
        img = sync_msgs['preview']
        dets = sync_msgs['dets']
        for i, det in enumerate(dets.detections):
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            # node.warn(f"Sending {i + 1}. age/gender det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            cfg.setResize(64, 64)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
""")

manip_manip = pipeline.create(dai.node.ImageManip)
manip_manip.initialConfig.setResize(64, 64)
manip_manip.inputConfig.setWaitForMessage(True)
image_manip_script.outputs['manip_cfg'].link(manip_manip.inputConfig)
image_manip_script.outputs['manip_img'].link(manip_manip.inputImage)

# This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
# face that was detected by the face-detection NN.
emotions_nn = pipeline.create(dai.node.NeuralNetwork)
emotions_nn.setBlobPath(blobconverter.from_zoo(name="emotions-recognition-retail-0003", shaves=6))
manip_manip.out.link(emotions_nn.input)

recognition_xout = pipeline.create(dai.node.XLinkOut)
recognition_xout.setStreamName("expression")
emotions_nn.out.link(recognition_xout.input)




with dai.Device(pipeline) as device:
    facerec = FaceRecognition(databases, args.name, name_to_id_mapping)
    sync = TwoStageHostSeqSync()
    text = TextHelper()  # Define the TextHelper object here

    queues = {}
    # Create output queues
    for name in ["color", "detection", "recognition","expression"]:
        queues[name] = device.getOutputQueue(name)

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections, and age/gender recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            expressions = msgs["expression"]                    

            for i, detection in enumerate(detections):
                
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                exp = expressions[i]

                emotion_results = np.array(exp.getFirstLayerFp16())
                emotion_name = emotions[np.argmax(emotion_results)]

                

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                y = (bbox[1] + bbox[3]) // 2
                cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

                features = np.array(msgs["recognition"][i].getFirstLayerFp16())
                conf, name = facerec.new_recognition(features)

    # Get assigned ID by name
                id = facerec.get_id_by_name(name)
                

    # Display the face recognition results
                text.putText(frame, f"{name}: ID _{id}_ {(100*conf):.0f}%", (bbox[0] + 10, bbox[1] + 35))
                facerec.detected_faces_info.append({"name": name, "id": id, "confidence": conf, "emotion": emotion_name})
                #current_time = datetime.now().timestamp()
                #data = {
                       # "id": id,
                        #"emotion": emotion_name,
                        #"timestamp": current_time,
                        #"camera ID":"oak camera"
                       # }
               # url = 'http://18.188.42.141:8080/face/detection/insert'
               # with open('token.txt', 'r') as file:
                        # token = file.read().strip()
                         #print(token)

                #headers = {
                        #'Authorization': 'Bearer '+token,
                        #'Content-Type': 'application/json'
                            #}
                #print(data)
                # response = requests.post(url, headers=headers, json=data)
                # if response.text:  # Check if response is not empty
                #       print(response.json())
                # else:
                #     print("Empty response received from the server.")


            

            # Save detected faces information to a file
            facerec.save_detected_faces_info()

            # Optionally clear the list if needed
            # facerec.clear_detected_faces_info()

            cv2.imshow("color", cv2.resize(frame, (800, 800)))

        if cv2.waitKey(1) == ord('q'):
            break



   

