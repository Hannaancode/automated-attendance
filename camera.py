import cv2

cap = cv2.VideoCapture("rtsp://admin:tvmCX123@192.168.29.194:554/Streaming/Channels/101/")

while True:
    _, frame = cap.read()
    cv2.imshow("RTSP", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
