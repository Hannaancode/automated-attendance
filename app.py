from flask import Flask, render_template
import tkinter as tk
from tkinter import ttk
from datetime import datetime

import json

app = Flask(__name__)

def read_detected_faces_info(file_path):
    data = []
    # Open the file and read its contents
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse each line to extract name, ID, and time
                info = json.loads(line)
                name = info.get("name", "N/A")
                face_id = info.get("id", "N/A")
                timestamp = info.get("timestamp", "N/A")
                emotion = info.get("emotion", "N/A")
                # Get the timestamp directly from the JSON data
                # Format timestamp string to YYYY-MM-DD HH:MM:SS
                formatted_timestamp = datetime.fromtimestamp(timestamp)
                date = formatted_timestamp.strftime('%Y-%m-%d')
                time = formatted_timestamp.strftime('%H:%M:%S')
                data.append((date, name, face_id, time, emotion))
            except json.JSONDecodeError:
                # Skip lines that cannot be parsed as JSON
                pass
    return data

@app.route('/')
def index():
    # Read the data from the file
    file_path = "C:/Users/abdul/depthai-python/examples/face recognition/data/detected_faces_info.txt"
    data = read_detected_faces_info(file_path)

    # Render the template with the data
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
