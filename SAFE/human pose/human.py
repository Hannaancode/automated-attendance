
import cv2
# coding=utf-8
import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np

import json
import time
from datetime import datetime

pipeline = Pipeline()

# Define the input stream
cam_rgb = pipeline.create_rgb_camera()
cam_rgb.set_preview_size(456, 256)

# Create the neural network node
pose_nn = pipeline.create_nn_node(model='models/human-pose-estimation-0001.blob', 
                                   model_type=Pipeline.NN_ModelType.blob, 
                                   shaves=6)

# Link the camera output to the neural network input
cam_rgb.preview.link(pose_nn.input)

# Define a function to process the output data
def process_output(output_data):
    # Extract and process keypoints from output data
    # This is a simplified example, you may need to adjust based on the actual output format
    keypoints = []
    # Process output_data to extract keypoints
    return keypoints

# Start the pipeline
with Camera(pipeline=pipeline) as cam:
    while True:
        # Get the next frame
        frame = cam.get_frame()
        if frame is not None:
            # Perform inference
            results = cam.get_model_results(pose_nn)
            if results is not None:
                keypoints = process_output(results)
                # Process keypoints or display them on the frame
                for point in keypoints:
                    x, y = point
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                # Display frame with keypoints
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cv2.destroyAllWindows()
