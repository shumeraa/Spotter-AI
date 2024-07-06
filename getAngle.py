from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load a pretrained YOLOv8m model
model = YOLO("yolov8m-pose.pt")


# Define a function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    # p1, p2, p3 are the points in format [x, y]
    # Calculate the vectors from p2 to p1 and p2 to p3
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Calculate the angle in radians between vectors v1 and v2 using the dot product and norms of the vectors
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# Open the video stream from a file
cap = cv2.VideoCapture("Data/KneeCave.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use a model to detect keypoints in the frame and convert them to numpy array
    results = model(frame)
    annotated_frame = results[0].plot()
    keypoints = results[0].keypoints.xyn.cpu().numpy()

    # Indices for right hip, knee, and ankle keypoints
    right_hip_index = 12
    right_knee_index = 14
    right_ankle_index = 16

    # Extract the coordinates for the right hip, knee, and ankle keypoints
    right_hip = np.squeeze(keypoints[:, right_hip_index, :2])
    right_knee = np.squeeze(keypoints[:, right_knee_index, :2])
    right_ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Convert normalized coordinates to pixel coordinates
    right_hip = (right_hip * np.array([frame.shape[1], frame.shape[0]])).astype(int)
    right_knee = (right_knee * np.array([frame.shape[1], frame.shape[0]])).astype(int)
    right_ankle = (right_ankle * np.array([frame.shape[1], frame.shape[0]])).astype(int)

    # Display the calculated knee angle on the frame
    cv2.putText(
        annotated_frame,
        f"Knee Angle: {knee_angle:.2f}",
        (int(right_knee[0]), int(right_knee[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    print(f"Knee Angle: {knee_angle:.2f}")

    # Display the frame with the annotated angle in a window
    cv2.imshow("Example", annotated_frame)

    # Introduce a delay between frames and break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()