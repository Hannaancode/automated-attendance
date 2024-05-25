import numpy as np
import cv2

# RTSP URL for Channel 0 of the camera
rtsp_url = "rtsp://admin:tvmCX123@192.168.29.194:554/Streaming/Channels/0/"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Error: Failed to receive frame.")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
