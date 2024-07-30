from ultralytics import YOLO
from analyzeAndPlot import (
    calculate_angle,
    plotLegAndKneeAngle,
    squatIsBelowParallel,
    plotRepCount,
    squatIsAtTheTop,
    checkKneeCollapse,
    plotShoulderLine,
)
import multiprocessing
from callLLM import callLLMs
import os
import numpy as np

missedDepthTuple = ("missedDepth", "The client missed depth on the squat")
leftKneeCaveTuple = (
    "rightKneeCave",
    "The client's left knee is caving in on the squat",
)
rightKneeCaveTuple = (
    "leftKneeCave",
    "The client's right knee is caving in on the squat",
)

# Indices for right hip, knee, and ankle keypoints
right_hip_index = 12
right_knee_index = 14
right_ankle_index = 16

left_hip_index = 11
left_knee_index = 13
left_ankle_index = 15

left_shouler_index = 5
right_shoulder_index = 6


class SquatAnalyzer:
    def __init__(self, recordingsFolder):
        self.model = YOLO("yolov8m-pose.pt")
        self.right_hip = None
        self.right_knee = None
        self.right_ankle = None
        self.right_knee_angle = None
        self.left_hip = None
        self.left_knee = None
        self.left_ankle = None
        self.left_knee_angle = None
        self.leftKneeCollapseRep = None
        self.rightKneeCollapseRep = None
        self.squatIsBelowParallel = None
        self.squatIsAtTheTop = None
        self.recordingsFolder = recordingsFolder
        self.squatWasBelowParallelLastFrame = False
        self.onFirstFrame = True
        self.squatRep = 0
        self.squatComingBackUp = False
        self.squatWasAtTopLastFrame = False
        self.squatComingDown = False
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.llmCall_process = multiprocessing.Process(
            target=llmCall_worker, args=(self.input_queue, self.output_queue)
        )
        self.llmCall_process.start()

    def emptyRecordingsFolder(self):
        for file in os.listdir(self.recordingsFolder):
            os.remove(os.path.join(self.recordingsFolder, file))

    def process_frame(self, frame):
        # Use a model to detect keypoints in the frame and convert them to numpy array
        results = self.model(frame)
        # annotated_frame = results[0].plot()  # Plot the keypoints on the frame
        keypoints = results[0].keypoints.xyn.cpu().numpy()

        self.calculateCoordinates(keypoints, frame)

        if self.left_knee_angle is None or self.right_knee_angle is None:
            return frame, False

        self.calculateSquatStatus()

        self.analyze_squat(keypoints)
        if self.squatRep != self.rightKneeCollapseRep and checkKneeCollapse(
            self.right_hip,
            self.right_knee,
            self.right_ankle,  # just for right leg
            self.left_knee,
        ):
            self.rightKneeCollapseRep = self.squatRep
            self.input_queue.put(
                (rightKneeCaveTuple[0], rightKneeCaveTuple[1], self.squatRep)
            )

        if self.squatRep != self.leftKneeCollapseRep and checkKneeCollapse(
            self.left_hip,
            self.left_knee,
            self.left_ankle,  # just for right leg
            self.right_knee,
        ):
            self.leftKneeCollapseRep = self.squatRep
            self.input_queue.put(
                (leftKneeCaveTuple[0], leftKneeCaveTuple[1], self.squatRep)
            )
        plotLegAndKneeAngle(
            frame,
            self.left_hip,
            self.left_ankle,
            self.left_knee,
            self.left_knee_angle,
            left=True,
        )
        plotLegAndKneeAngle(
            frame,
            self.right_hip,
            self.right_ankle,
            self.right_knee,
            self.right_knee_angle,
            left=False,
        )

        plotRepCount(frame, self.squatRep)
        # self.calculateShoulderCoordinatesAndPlot(keypoints, frame)

        return frame, True

    def analyze_squat(self, keypoints):
        if self.onFirstFrame:
            self.onFirstFrame = False
            self.squatWasBelowParallelLastFrame = self.squatIsBelowParallel
        else:
            if self.squatComingDown and self.squatIsAtTheTop:
                self.squatComingDown = False
                self.input_queue.put(
                    (missedDepthTuple[0], missedDepthTuple[1], self.squatRep)
                )

            if self.squatComingBackUp:
                self.squatComingDown = False
            if self.squatWasBelowParallelLastFrame and not self.squatIsBelowParallel:
                self.squatComingBackUp = True
            if self.squatWasAtTopLastFrame and not self.squatIsAtTheTop:
                self.squatComingDown = True
            if self.squatComingBackUp and self.squatIsAtTheTop:
                self.squatRep += 1
                self.squatComingBackUp = False

            self.squatWasBelowParallelLastFrame = self.squatIsBelowParallel
            self.squatWasAtTopLastFrame = self.squatIsAtTheTop

    def calculateCoordinates(self, keypoints, frame):
        right_hip_norm = np.squeeze(keypoints[:, right_hip_index, :2])
        right_knee_norm = np.squeeze(keypoints[:, right_knee_index, :2])
        right_ankle_norm = np.squeeze(keypoints[:, right_ankle_index, :2])

        if (
            right_hip_norm.shape != (2,)
            or right_knee_norm.shape != (2,)
            or right_ankle_norm.shape != (2,)
        ):
            print("Some keypoints are missing")
            self.right_hip = None
            self.right_knee = None
            self.right_ankle = None
            self.right_knee_angle = None
        else:
            self.right_hip = (
                (right_hip_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            self.right_knee = (
                (right_knee_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            self.right_ankle = (
                (right_ankle_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            self.right_knee_angle = calculate_angle(
                self.right_ankle, self.right_knee, self.right_hip
            )

        left_hip_norm = np.squeeze(keypoints[:, left_hip_index, :2])
        left_knee_norm = np.squeeze(keypoints[:, left_knee_index, :2])
        left_ankle_norm = np.squeeze(keypoints[:, left_ankle_index, :2])

        if (
            left_hip_norm.shape != (2,)
            or left_knee_norm.shape != (2,)
            or left_ankle_norm.shape != (2,)
        ):
            print("Some keypoints are missing")
            self.left_hip = None
            self.left_knee = None
            self.left_ankle = None
            self.left_knee_angle = None
        else:
            self.left_hip = (
                (left_hip_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            self.left_knee = (
                (left_knee_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            self.left_ankle = (
                (left_ankle_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            self.left_knee_angle = calculate_angle(
                self.left_ankle, self.left_knee, self.left_hip
            )

    def calculateShoulderCoordinatesAndPlot(self, keypoints, frame):
        left_shoulder_norm = np.squeeze(keypoints[:, left_shouler_index, :2])
        right_shoulder_norm = np.squeeze(keypoints[:, right_shoulder_index, :2])

        if left_shoulder_norm.shape != (2,) or right_shoulder_norm.shape != (2,):
            print("Some keypoints are missing")
        else:
            left_shoulder = (
                (left_shoulder_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )
            right_shoulder = (
                (right_shoulder_norm * np.array([frame.shape[1], frame.shape[0]]))
                .astype(int)
                .tolist()
            )

            plotShoulderLine(frame, left_shoulder, right_shoulder)

    def calculateSquatStatus(self):
        if self.left_knee_angle and self.right_knee_angle:
            self.squatIsBelowParallel = squatIsBelowParallel(
                self.left_knee_angle, self.right_knee_angle
            )
            self.squatIsAtTheTop = squatIsAtTheTop(
                self.left_knee_angle, self.right_knee_angle
            )
        else:
            self.squatIsBelowParallel = None
            self.squatIsAtTheTop = None

    def cleanup(self):
        self.input_queue.put(None)
        self.llmCall_process.join()


def llmCall_worker(input_queue, output_queue):
    while True:
        message_rep_Tuple = input_queue.get()
        if message_rep_Tuple is None:  # Sentinel value to signal the end of the process
            break
        callLLMs(message_rep_Tuple)
        print("LLM called")
        output_queue.put(f"Processed: {message_rep_Tuple}")
