import tkinter as tk
import subprocess
import threading
import os
import time
import datetime

def run_clarity_periodically():
    while True:
        # Run clarity.py
        os.system("python clarity.py")

       
        time.sleep(15)

# Start a thread to run the clarity.py script every 10 seconds
clarity_thread = threading.Thread(target=run_clarity_periodically)
clarity_thread.start()


def run_radio():
    subprocess.Popen(['python', 'new radio.py'])

def run_new_radio():
    subprocess.Popen(['python', 'radio.py'])

# Create the main window
root = tk.Tk()
root.title("Video Player")
root.configure(bg="black")  # Set background color to black
root.geometry("300x150")  # Larger window size

# Create buttons to run each program with some styling
radio_button = tk.Button(root, text="Add New Face", command=run_radio, bg="blue", fg="white", width=15)
radio_button.pack(pady=5)  # Add space between buttons

new_radio_button = tk.Button(root, text="Run Video File", command=run_new_radio, bg="blue", fg="white", width=15)
new_radio_button.pack(pady=5)  # Add space between buttons

root.mainloop()
