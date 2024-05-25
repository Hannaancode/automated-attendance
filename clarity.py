import cv2
import os
from openvino.inference_engine import IECore

def load_model(model_xml, model_bin):
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU", num_requests=1)
    return exec_net

def face_detection(image_path, exec_net):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image {image_path}")
        return False

    input_blob = next(iter(exec_net.input_info))
    output_blob = next(iter(exec_net.outputs))

    # Preprocess the image
    input_image = cv2.resize(image, (300, 300))  # Model input shape is 300x300
    input_image = input_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    input_image = input_image.reshape((1, 3, 300, 300))  # Add batch size

    # Perform inference
    exec_net.start_async(request_id=0, inputs={input_blob: input_image})
    if exec_net.requests[0].wait(-1) == 0:
        # Get the inference request
        request = exec_net.requests[0]

        # Get the outputs
        outputs = request.output_blobs

        # Parse the outputs
        output = outputs[next(iter(outputs))]
        confidence = output.buffer[0][0][0][2]
        print(f"Confidence level for {image_path}: {confidence}")

        if confidence < 0.70:
            os.remove(image_path)
            print(f"Deleted {image_path} because face detection confidence was below 0.7")
            return True
    return False


folder_path = "./face_database/unknown_faces"

# Path to the directory containing the model files
model_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current Python file

# Path to the model files
model_xml = os.path.join(model_dir, "face-detection-retail-0004.xml")
model_bin = os.path.join(model_dir, "face-detection-retail-0004.bin")

# Load the model
exec_net = load_model(model_xml, model_bin)

# Loop through all images in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if face_detection(file_path, exec_net):
        continue  # Skip to the next image if image was deleted
