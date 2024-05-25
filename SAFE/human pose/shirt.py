import cv2
import numpy as np
import webcolors

def capture_shirt_color(frame, bbox):
    # Extract the bounding box coordinates
    xmin, ymin, xmax, ymax = bbox

    # Crop the frame to the bounding box
    shirt_roi = frame[ymin:ymax, xmin:xmax]

    # Calculate the average color in the ROI
    avg_color = np.mean(shirt_roi, axis=(0, 1)).astype(int)

    # Convert the average color to BGR format (OpenCV format)
    avg_color_bgr = avg_color[::-1]

    return avg_color_bgr

def convert_bgr_to_color_name(bgr_color):
    # Convert the BGR color to RGB format
    rgb_color = tuple(reversed(bgr_color))

    try:
        # Get the closest color name from the RGB color
        closest_color = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        closest_color = "Unknown"

    return closest_color


def main():
    # Open a video capture device (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Define a bounding box for the shirt area (manually or using a face detection algorithm)
        # For demonstration, manually set a bounding box around the shirt area
        bbox = (100, 100, 300, 300)  # Format: (xmin, ymin, xmax, ymax)

        # Capture the shirt color
        shirt_color = capture_shirt_color(frame, bbox)
        print("Shirt Color (BGR):", shirt_color)

        # Convert the BGR color to a human-readable color name
        color_name = convert_bgr_to_color_name(shirt_color)
        print("Shirt Color (Name):", color_name)

        # Display the color name on the frame
        cv2.putText(frame, f"Shirt Color: {color_name}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Wait for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
