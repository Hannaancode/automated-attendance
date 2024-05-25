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
                timestamp = info.get("timestamp", "N/A")  # Get the timestamp directly from the JSON data
                # Format timestamp string to YYYY-MM-DD HH:MM:SS
                formatted_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                data.append((name, face_id, formatted_timestamp))
            except json.JSONDecodeError:
                # Skip lines that cannot be parsed as JSON
                pass
    return data

def browse_file():
    file_path = "./data/detected_faces_info.txt"
    if file_path:
        # Read the contents of the selected file
        data = read_detected_faces_info(file_path)
        # Clear the existing treeview content
        tree.delete(*tree.get_children())
        
        # Dictionary to keep track of total work hours for each person per day
        total_work_hours = {}
        
        for name, face_id, timestamp in data:
            # Extract date from timestamp
            date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').date()
            # Check if the name exists in the total work hours dictionary for the current date
            if (date, name) not in total_work_hours:
                # If not, initialize the total work hours for that person on that day
                total_work_hours[(date, name)] = {"Entry": None, "Exit": None, "Total Work Hours": None}
            # If the entry time is not set, set it as the current timestamp
            if total_work_hours[(date, name)]["Entry"] is None:
                total_work_hours[(date, name)]["Entry"] = timestamp
            # If the entry time is set but the exit time is not set, set it as the current timestamp
            elif total_work_hours[(date, name)]["Exit"] is None:
                total_work_hours[(date, name)]["Exit"] = timestamp
                # Calculate the duration between entry and exit times
                entry_time = datetime.strptime(total_work_hours[(date, name)]["Entry"], '%Y-%m-%d %H:%M:%S')
                exit_time = datetime.strptime(total_work_hours[(date, name)]["Exit"], '%Y-%m-%d %H:%M:%S')
                duration = exit_time - entry_time
                total_work_hours[(date, name)]["Total Work Hours"] = str(duration)
                # Insert the data into the treeview
                tree.insert("", tk.END, values=[date, name, face_id, total_work_hours[(date, name)]["Entry"], 
                                                 total_work_hours[(date, name)]["Exit"], 
                                                 total_work_hours[(date, name)]["Total Work Hours"]])
                # Reset entry and exit times
                total_work_hours[(date, name)]["Entry"] = None
                total_work_hours[(date, name)]["Exit"] = None

def calculate_total_attendance():
    total_attendance_data = {}
    for child in tree.get_children():
        values = tree.item(child)["values"]
        name = values[1]
        date = values[0]
        total_work_hours = values[5]
        if (name, date) not in total_attendance_data:
            total_attendance_data[(name, date)] = 0
        if total_work_hours:
            hours, minutes, seconds = map(int, total_work_hours.split(":"))
            total_minutes = hours * 60 + minutes + seconds / 60
            total_attendance_data[(name, date)] += total_minutes

    # Create a new window to display total attendance data
    total_attendance_window = tk.Toplevel(root)
    total_attendance_window.title("Total Attendance")

    # Create a treeview widget for total attendance
    total_attendance_tree = ttk.Treeview(total_attendance_window, columns=("Name", "Date", "Total Work Hours"))

    # Define column headings
    total_attendance_tree.heading("#0", text="", anchor=tk.W)
    total_attendance_tree.heading("Name", text="Name", anchor=tk.W)
    total_attendance_tree.heading("Date", text="Date", anchor=tk.W)
    total_attendance_tree.heading("Total Work Hours", text="Total Work Hours", anchor=tk.W)

    # Define column widths
    total_attendance_tree.column("#0", width=0, stretch=tk.NO)
    total_attendance_tree.column("Name", width=100, stretch=tk.NO)
    total_attendance_tree.column("Date", width=100, stretch=tk.NO)
    total_attendance_tree.column("Total Work Hours", width=150, stretch=tk.NO)

    # Insert total attendance data into the treeview
    for (name, date), total_minutes in total_attendance_data.items():
        hours = int(total_minutes / 60)
        minutes = int(total_minutes % 60)
        total_work_hours = f"{hours:02d}:{minutes:02d}"
        total_attendance_tree.insert("", tk.END, values=[name, date, total_work_hours])

    # Add treeview to total attendance window
    total_attendance_tree.pack(expand=True, fill=tk.BOTH)

# Create main window
root = tk.Tk()
root.title("Detected Faces Info Reader")

# Create a treeview widget
tree = ttk.Treeview(root, columns=("Date", "Name", "ID", "Entry Time", "Exit Time", "Total Working Hours"))

# Define column headings
tree.heading("#0", text="", anchor=tk.W)
tree.heading("Date", text="Date", anchor=tk.W)
tree.heading("Name", text="Name", anchor=tk.W)
tree.heading("ID", text="ID", anchor=tk.W)
tree.heading("Entry Time", text="Entry Time", anchor=tk.W)
tree.heading("Exit Time", text="Exit Time", anchor=tk.W)
tree.heading("Total Working Hours", text="Total Working Hours", anchor=tk.W)

# Define column widths
tree.column("#0", width=0, stretch=tk.NO)
tree.column("Date", width=100, stretch=tk.NO)
tree.column("Name", width=100, stretch=tk.NO)
tree.column("ID", width=100, stretch=tk.NO)
tree.column("Entry Time", width=150, stretch=tk.NO)
tree.column("Exit Time", width=150, stretch=tk.NO)
tree.column("Total Working Hours", width=150, stretch=tk.NO)

# Add treeview to main window
tree.pack(expand=True, fill=tk.BOTH)

# Automatically load data when the GUI starts
browse_file()

# Create a button to calculate total attendance
total_attendance_button = tk.Button(root, text="Total Attendance", command=calculate_total_attendance)
total_attendance_button.pack(pady=10)

# Start the GUI main loop
root.mainloop()
