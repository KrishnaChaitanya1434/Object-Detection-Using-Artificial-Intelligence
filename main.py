import cv2
import numpy as np

# Create a cascade classifier object
face_cascade = cv2.CascadeClassifier(r"C:\Users\Admin\PycharmProjects\detection4(25-03)\venv\Lib\site-packages\cv2\facedetection_haarcascade_frontalface_default.xml at master · adarsh1021_facedetection · GitHub.html\haarcascade_frontalface_default.xml")

# Capture frames from a video stream or a camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream or camera
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray)

    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Frame', frame)

    # Stop the program if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()