import cv2
import numpy as np

# Load video
video = cv2.VideoCapture('input_video.mp4')

# Create background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Initialize counters
people_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Apply background subtraction
    mask = background_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter small objects
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            people_count += 1

    # Display the count on the video
    cv2.putText(frame, f'Count: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('People Counter', frame)

    # Quit with 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
