import numpy as np
import matplotlib.pyplot as plt
import cv2


# Indices for right hip, knee, and ankle keypoints
right_hip_index = 12
right_knee_index = 14
right_ankle_index = 16

left_hip_index = 11
left_knee_index = 13
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


def squatIsBelowParallel(keypoints):
    # Extract the coordinates for the right hip, knee, and ankle keypoints
    right_hip = np.squeeze(keypoints[:, right_hip_index, :2])
    right_knee = np.squeeze(keypoints[:, right_knee_index, :2])
    right_ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    if right_hip.shape != (2,) or right_knee.shape != (2,) or right_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return None

    # Extract the coordinates for the left hip, knee, and ankle keypoints
    left_hip = np.squeeze(keypoints[:, left_hip_index, :2])
    left_knee = np.squeeze(keypoints[:, left_knee_index, :2])
    left_ankle = np.squeeze(keypoints[:, left_ankle_index, :2])

    if left_hip.shape != (2,) or left_knee.shape != (2,) or left_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return None

    left_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    if left_knee_angle < 110 and right_knee_angle < 110:
        # in lower portion of squat
        return True
    else:
        # in upper portion of squat
        return False


def squatIsAtTheTop(keypoints):
    # Extract the coordinates for the right hip, knee, and ankle keypoints
    right_hip = np.squeeze(keypoints[:, right_hip_index, :2])
    right_knee = np.squeeze(keypoints[:, right_knee_index, :2])
    right_ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    if right_hip.shape != (2,) or right_knee.shape != (2,) or right_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return None

    # Extract the coordinates for the left hip, knee, and ankle keypoints
    left_hip = np.squeeze(keypoints[:, left_hip_index, :2])
    left_knee = np.squeeze(keypoints[:, left_knee_index, :2])
    left_ankle = np.squeeze(keypoints[:, left_ankle_index, :2])

    if left_hip.shape != (2,) or left_knee.shape != (2,) or left_ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return None

    left_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    right_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    if left_knee_angle > 170 and right_knee_angle > 170:
        # at the top of the squat
        return True
    else:
        # not at the top of the sqaut
        return False


def getKneeAngle(frame, keypoints, left):
    # Extract the coordinates for the right hip, knee, and ankle keypoints
    if left:
        hip = np.squeeze(keypoints[:, left_hip_index, :2])
        knee = np.squeeze(keypoints[:, left_knee_index, :2])
        ankle = np.squeeze(keypoints[:, left_ankle_index, :2])
    else:
        hip = np.squeeze(keypoints[:, right_hip_index, :2])
        knee = np.squeeze(keypoints[:, right_knee_index, :2])
        ankle = np.squeeze(keypoints[:, right_ankle_index, :2])

    if hip.shape != (2,) or knee.shape != (2,) or ankle.shape != (2,):
        print("Some keypoints are missing, cannot calculate the angle")
        return None

    knee_angle = calculate_angle(hip, knee, ankle)
    # print(f"Knee Angle: {knee_angle}")
    return knee_angle


def plotKneeAngle(frame, keypoints, knee_angle, left):
    xPixels = 0
    if left:
        # Extract the coordinates for the left knee keypoint
        knee = np.squeeze(keypoints[:, left_knee_index, :2])
    else:
        # Extract the coordinates for the right knee keypoint
        knee = np.squeeze(keypoints[:, right_knee_index, :2])
        xPixels = 130

    # Convert normalized coordinates to pixel coordinates
    knee_pixel = knee * np.array([frame.shape[1], frame.shape[0]]).astype(int)

    # Display the calculated knee angle on the frame
    cv2.putText(
        frame,
        f"Knee Angle: {knee_angle:.2f}",
        (int(knee_pixel[0]) - xPixels, int(knee_pixel[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    # print(f"Knee Angle: {knee_angle:.2f}")
    return True


def plotRepCount(frame, repCount):
    cv2.putText(
        frame,
        f"Rep Count: {repCount}",
        (10, 50),  # Position in the top left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Larger font size
        (255, 0, 0),  # Black color
        2,
        cv2.LINE_AA,
    )
