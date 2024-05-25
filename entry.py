import tkinter as tk
from tkinter import ttk
from datetime import datetime
import json

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

def browse_file():
    file_path = "C:/Users/abdul/depthai-python/examples/face recognition/data/detected_faces_info.txt"
    if file_path:
        # Read the contents of the selected file
        data = read_detected_faces_info(file_path)
        # Clear the existing treeview content
        tree.delete(*tree.get_children())
        
        for date, name, face_id, time, emotion in data:
            # Insert the data into the treeview
            tree.insert("", tk.END, values=[date, name, face_id, time, emotion])

# Create main window
root = tk.Tk()
root.title("Detected Faces Info Reader")

# Create a treeview widget
tree = ttk.Treeview(root, columns=("Date", "Name", "ID", "Entry Time", "Emotion"))

# Define column headings
tree.heading("#0", text="", anchor=tk.W)
tree.heading("Date", text="Date", anchor=tk.W)
tree.heading("Name", text="Name", anchor=tk.W)
tree.heading("ID", text="ID", anchor=tk.W)
tree.heading("Entry Time", text="Entry Time", anchor=tk.W)
tree.heading("Emotion", text="Emotion", anchor=tk.W)

# Define column widths
tree.column("#0", width=0, stretch=tk.NO)
tree.column("Date", width=100, stretch=tk.NO)
tree.column("Name", width=100, stretch=tk.NO)
tree.column("ID", width=100, stretch=tk.NO)
tree.column("Entry Time", width=150, stretch=tk.NO)
tree.column("Emotion", width=150, stretch=tk.NO)

# Add treeview to main window
tree.pack(expand=True, fill=tk.BOTH)

# Automatically load data when the GUI starts
browse_file()

# Start the GUI main loop
root.mainloop()
