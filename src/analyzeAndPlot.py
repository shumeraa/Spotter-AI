import numpy as np
import cv2


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


def squatIsBelowParallel(left_knee_angle, right_knee_angle):
    if left_knee_angle < 110 and right_knee_angle < 110:
        # in lower portion of squat
        return True
    else:
        # in upper portion of squat
        return False


def squatIsAtTheTop(left_knee_angle, right_knee_angle):

    if left_knee_angle > 170 and right_knee_angle > 170:
        # at the top of the squat
        return True
    else:
        # not at the top of the sqaut
        return False


def plotLegAndKneeAngle(frame, hip, ankle, knee, knee_angle, left):

    if not hip or not ankle or not knee:
        return False

    xPixels = 0
    if left:
        xPixels = 0
    else:
        xPixels = 130

    point_color = (0, 0, 0)  # Black for points
    line_color = (0, 255, 0)  # Red for lines
    point_radius = 7  # Radius for the points
    line_thickness = 2  # Thickness for the lines

    # Draw lines connecting knee to hip and knee to ankle
    cv2.line(frame, knee, hip, line_color, line_thickness)
    cv2.line(frame, knee, ankle, line_color, line_thickness)

    # Draw points at the specified locations
    cv2.circle(frame, knee, point_radius, point_color, -1)
    cv2.circle(frame, hip, point_radius, point_color, -1)
    cv2.circle(frame, ankle, point_radius, point_color, -1)

    # Display the calculated knee angle on the frame
    cv2.putText(
        frame,
        f"Knee Angle: {knee_angle:.2f}",
        (int(knee[0]) - xPixels, int(knee[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    # print(f"Knee Angle: {knee_angle:.2f}")
    return True


def plotShoulderLine(frame, leftShoulder, rightShoulder):
    x1, y1 = leftShoulder
    x2, y2 = rightShoulder

    if x2 - x1 == 0:
        raise ValueError("The line is vertical, slope is undefined.")

    slope = (y2 - y1) / (x2 - x1)
    print(f"Slope is: {slope}")

    line_color = (0, 255, 0)  # Red for lines
    line_thickness = 2  # Thickness for the lines

    # Draw lines connecting knee to hip and knee to ankle
    cv2.line(frame, leftShoulder, rightShoulder, line_color, line_thickness)
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


def checkKneeCollapse(hip, knee, ankle, opposite_knee, threshold=0):
    # Unpack the coordinates
    x1, y1 = hip
    x2, y2 = ankle
    xk, yk = knee
    xk_other, yk_other = opposite_knee

    # Calculate the slope (m) and y-intercept (b) of the line y = mx + b
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Determine the "in" side based on other_knee
        y_on_line_other = m * xk_other + b
        in_side = "right" if yk_other > y_on_line_other else "left"

        # Check if knee is on the "in" side
        y_on_line_knee = m * xk + b
        is_knee_in = (yk > y_on_line_knee and in_side == "right") or (
            yk <= y_on_line_knee and in_side == "left"
        )
    else:
        m = None  # Line is vertical
        b = x1  # x = b is the equation of the vertical line

        # Determine the "in" side based on other_knee
        in_side = "right" if xk_other > b else "left"

        # Check if knee is on the "in" side
        is_knee_in = (xk > b and in_side == "right") or (xk <= b and in_side == "left")

    # print(f"Caving = {is_knee_in}")
    return is_knee_in
