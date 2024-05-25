import os
import numpy as np
import cv2

# Path to the 'database' folder containing npz files
database_folder = 'databases'

# Get a list of all npz files in the 'database' folder
npz_files = [f for f in os.listdir(database_folder) if f.endswith('.npz')]

# Iterate over each npz file
for npz_file in npz_files:
    # Load the npz file
    with np.load(os.path.join(database_folder, npz_file)) as data:
        # Extract the image array
        image = data['image']

        # Save the image as a separate file
        image_path = os.path.join('output_images', f'{npz_file[:-4]}.jpg')
        cv2.imwrite(image_path, image)

        print(f"Extracted and saved image from {npz_file}")
