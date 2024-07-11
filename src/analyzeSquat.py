import numpy as np
import matplotlib.pyplot as plt
import cv2


# Indices for right hip, knee, and ankle keypoints
right_hip_index = 12
right_knee_index = 14
right_ankle_index = 16

left_hip_index = 11
left_knee_index = 14
left_ankle_index = 15


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


def countSquats(frame, keypoints):
    # Extract the coordinates for the right hip, knee, and ankle keypoints
    right_hip = np.squeeze(keypoints[:, right_hip_index, :2])
    right_knee = np.squeeze(keypoints[:, right_knee_index, :2])
    right_ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    if right_hip.shape != (2,) or right_knee.shape != (2,) or right_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return False

    # Extract the coordinates for the right hip, knee, and ankle keypoints
    left_hip = np.squeeze(keypoints[:, left_hip_index, :2])
    left_knee = np.squeeze(keypoints[:, left_knee_index, :2])
    left_ankle = np.squeeze(keypoints[:, left_ankle_index, :2])

    if left_hip.shape != (2,) or left_knee.shape != (2,) or left_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return False

    left_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    if left_knee_angle > 160 and right_knee_angle > 160:
        return True
    else:
        return False


def getKneeAngle(frame, keypoints):
    # Extract the coordinates for the right hip, knee, and ankle keypoints
    right_hip = np.squeeze(keypoints[:, right_hip_index, :2])
    right_knee = np.squeeze(keypoints[:, right_knee_index, :2])
    right_ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    if right_hip.shape != (2,) or right_knee.shape != (2,) or right_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return None

    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    print(f"Knee Angle: {knee_angle}")
    return knee_angle


def plotKneeAngle(frame, keypoints, knee_angle):
    # Extract the coordinates for the right hip, knee, and ankle keypoints
    right_hip = np.squeeze(keypoints[:, right_hip_index, :2])
    right_knee = np.squeeze(keypoints[:, right_knee_index, :2])
    right_ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    # Convert normalized coordinates to pixel coordinates
    right_hip = (right_hip * np.array([frame.shape[1], frame.shape[0]])).astype(int)
    right_knee = (right_knee * np.array([frame.shape[1], frame.shape[0]])).astype(int)
    right_ankle = (right_ankle * np.array([frame.shape[1], frame.shape[0]])).astype(int)

    # Display the calculated knee angle on the frame
    cv2.putText(
        frame,
        f"Knee Angle: {knee_angle:.2f}",
        (int(right_knee[0]), int(right_knee[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    print(f"Knee Angle: {knee_angle:.2f}")
    return True
