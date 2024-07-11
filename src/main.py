from ultralytics import YOLO
import cv2
from analyzeSquat import getKneeAngle, plotKneeAngle

# load a pretrained YOLOv8m model
model = YOLO("yolov8m-pose.pt")


# Open the video stream from a file
cap = cv2.VideoCapture("Data\Squat.mp4")
cv2.namedWindow("Example", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Example", 1280, 720)
count = 0
playVideo = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use a model to detect keypoints in the frame and convert them to numpy array
    results = model(frame)
    annotated_frame = results[0].plot()
    keypoints = results[0].keypoints.xyn.cpu().numpy()

    knee_angle = getKneeAngle(annotated_frame, keypoints)
    if knee_angle is None:
        count += 1
    else:
        plotKneeAngle(annotated_frame, keypoints, knee_angle)

    if knee_angle and abs(knee_angle - 90) < 10:
        playVideo = 0

    # Display the frame with the annotated angle in a window
    cv2.imshow("Example", annotated_frame)

    # Introduce a delay between frames and break the loop if 'q' is pressed
    if cv2.waitKey(playVideo) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
print(f"Number of frames with missing keypoints: {count}")
cap.release()
cv2.destroyAllWindows()
