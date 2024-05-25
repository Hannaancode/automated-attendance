import cv2
import numpy as np
import os
import pickle
from openvino.inference_engine import IECore
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

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

# Global variables for GUI elements
root = tk.Tk()
root.title("Face Embedding Tool")
label = None  # Placeholder for the label
current_image_index = 0  # Variable to keep track of the current image index
images = []  # List to store image paths
unknown_faces_dir = 'face_database/unknown_faces'

def show_image(index, label):
    global current_image_index
    current_image_index = index
    image_name = images[index]
    image_path = os.path.join(unknown_faces_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.thumbnail((150, 150))
    photo = ImageTk.PhotoImage(image)

    label.config(image=photo)
    label.image = photo

def add_to_existing_faces():
    global images, current_image_index
    unknown_faces_dir = 'face_database/unknown_faces'
    images = os.listdir(unknown_faces_dir)

    if not images:
       tk.messagebox.showinfo("No Images Found", "No images found in the folder.")
       return

    # Create a new window to display the images
    add_window = tk.Toplevel(root)
    add_window.title("Add to Existing Faces")

    # Initialize variables for image navigation
    current_image_index = 0
    num_images = len(images)

    # Create a label for the image
    label = tk.Label(add_window)
    label.pack()

    show_image(current_image_index, label)

    def add_image_to_database():
        global current_image_index
        database_name = selected_database.get()
        if database_name:
            name = f"{input_name.get()}_{input_id.get()}"
            process_image(os.path.join(unknown_faces_dir, images[current_image_index]), name, database_name)
            os.remove(os.path.join(unknown_faces_dir, images[current_image_index]))
            current_image_index += 1
            if current_image_index < num_images:
                show_image(current_image_index, label)
            else:
                # Reload the list of databases
                databases = [f for f in os.listdir('face_database') if f.endswith('.pkl')]
                selected_database.set(databases[0] if databases else "")
                add_window.destroy()
    def delete_image():
        global images, num_images, current_image_index
        num_images = len(images)  # Update the number of images
        if num_images == 0:
            add_window.destroy()  # Close the tab if there are no images left
            return

        os.remove(os.path.join(unknown_faces_dir, images[current_image_index]))
        images = os.listdir(unknown_faces_dir)
        num_images = len(images)
        if num_images == 0:
            add_window.destroy()  # Close the tab if there are no images left
            return

        if current_image_index >= num_images:
            current_image_index = 0
        show_image(current_image_index, label)

    # Global variable to keep track of the number of images
    num_images = 0

# Rest of your code...


    # Create a button to delete the current image
    delete_button = ttk.Button(add_window, text="Delete Image", command=delete_image)
    delete_button.pack()           

    # Create buttons for image navigation
    prev_button = ttk.Button(add_window, text="Previous", command=lambda: show_image((current_image_index - 1) % num_images, label))
    prev_button.pack(side=tk.LEFT)

    next_button = ttk.Button(add_window, text="Next", command=lambda: show_image((current_image_index + 1) % num_images, label))
    next_button.pack(side=tk.RIGHT)

    # Create a frame for database selection
    database_frame = tk.Frame(add_window)
    database_frame.pack()

    # Label for database selection
    database_label = ttk.Label(database_frame, text="Select Database:")
    database_label.pack(side=tk.LEFT)

    # Get a list of existing databases
    databases = [f for f in os.listdir('face_database') if f.endswith('.pkl')]

    # Create a variable to store the selected database
    selected_database = tk.StringVar(add_window)
    selected_database.set(databases[0] if databases else "")

    # Create a dropdown menu for database selection
    database_menu = ttk.OptionMenu(database_frame, selected_database, *databases)
    database_menu.pack(side=tk.RIGHT)

    # Create a button to add the image to the selected database
    add_button = ttk.Button(add_window, text="Add to Database", command=add_image_to_database)
    add_button.pack()

    def add_new_face():
        name_window = tk.Toplevel(add_window)
        name_window.title("Enter Name and ID")
        

        def create_new_database():
            global current_image_index
            name = f"{new_name.get()}_{new_id.get()}"
            process_image(os.path.join(unknown_faces_dir, images[current_image_index]), name, f"{name}.pkl")
            os.remove(os.path.join(unknown_faces_dir, images[current_image_index]))
            current_image_index += 1
            if current_image_index < num_images:
                show_image(current_image_index, label)
            else:
                # Reload the list of databases
                databases = [f for f in os.listdir('face_database') if f.endswith('.pkl')]
                selected_database.set(databases[0] if databases else "")
                add_window.destroy()
            name_window.destroy()

        new_name_frame = tk.Frame(name_window)
        new_name_frame.pack()

        new_name_label = ttk.Label(new_name_frame, text="Enter Name:")
        new_name_label.pack(side=tk.LEFT)
        new_name = ttk.Entry(new_name_frame)
        new_name.pack(side=tk.LEFT)

        new_id_frame = tk.Frame(name_window)
        new_id_frame.pack()

        new_id_label = ttk.Label(new_id_frame, text="Enter ID:")
        new_id_label.pack(side=tk.LEFT)
        new_id = ttk.Entry(new_id_frame)
        new_id.pack(side=tk.LEFT)

        create_button = ttk.Button(name_window, text="Create Database", command=create_new_database)
        create_button.pack()

    new_face_button = ttk.Button(add_window, text="Add New Face (Create New Database)", command=add_new_face)
    new_face_button.pack()

    add_window.mainloop()

def process_image(image_path, name, database_name):
    global face_detection_exec_net, face_recognition_exec_net

    # Load the image
    frame = cv2.imread(image_path)

    # Initialize a list to store face embeddings
    face_data_128 = []

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
                face_data_128.append(embeddings)

    # Check if the pickle file already exists
    merged_name_id = f"{name}.pkl"
    pkl_file = os.path.join('face_database', database_name)
    if os.path.exists(pkl_file):
        # If the file exists, load the existing embeddings and append the new embeddings
        with open(pkl_file, 'rb') as f:
            existing_data = pickle.load(f)
        existing_data.extend(face_data_128)
        face_data_128 = existing_data

    # Save the updated embeddings in the pickle file
    with open(pkl_file, 'wb') as f:
        pickle.dump(face_data_128, f)

    cv2.destroyAllWindows()

def browse_image():
    filename = filedialog.askopenfilename(initialdir="/", title="Select an Image",
                                          filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"),("JPEG files", "*.jpeg")))
    if filename:
        name = f"{input_name.get()}_{input_id.get()}"
        process_image(filename, f"{input_name.get()}_{input_id.get()}", f"{input_name.get()}_{input_id.get()}.pkl")


add_button = ttk.Button(root, text="Add to Existing Faces", command=add_to_existing_faces)
add_button.pack()

# Create a label and an entry for entering the name
label_name = ttk.Label(root, text="Enter Name:")
label_name.pack()
input_name = ttk.Entry(root)
input_name.pack()

# Create a label and an entry for entering the ID
label_id = ttk.Label(root, text="Enter ID:")
label_id.pack()
input_id = ttk.Entry(root)
input_id.pack()

# Create a button to browse for an image
browse_button = ttk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack()

root.mainloop()
