from ultralytics import YOLO
from analyzeAndPlot import (
    getKneeAngle,
    plotLegAndKneeAngle,
    squatIsBelowParallel,
    plotRepCount,
    squatIsAtTheTop,
)
import multiprocessing
from callLLM import callLLMs
import os

missedDepthTuple = ("missedDepth", "The client missed depth on the squat")


class SquatAnalyzer:
    def __init__(self, recordingsFolder):
        self.model = YOLO("yolov8m-pose.pt")
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
        annotated_frame = frame
        keypoints = results[0].keypoints.xyn.cpu().numpy()

        left_knee_angle = getKneeAngle(annotated_frame, keypoints, left=True)
        right_knee_angle = getKneeAngle(annotated_frame, keypoints, left=False)

        if left_knee_angle is None or right_knee_angle is None:
            return annotated_frame, False

        self.analyze_squat(keypoints)
        plotLegAndKneeAngle(annotated_frame, keypoints, left_knee_angle, left=True)
        plotLegAndKneeAngle(annotated_frame, keypoints, right_knee_angle, left=False)
        plotRepCount(annotated_frame, self.squatRep)

        return annotated_frame, True

    def analyze_squat(self, keypoints):
        if self.onFirstFrame:
            self.onFirstFrame = False
            self.squatWasBelowParallelLastFrame = squatIsBelowParallel(keypoints)
        else:
            if self.squatComingDown and squatIsAtTheTop(keypoints):
                self.squatComingDown = False
                self.input_queue.put(
                    (missedDepthTuple[0], missedDepthTuple[1], self.squatRep)
                )

            if self.squatComingBackUp:
                self.squatComingDown = False
            elif self.squatWasBelowParallelLastFrame and not squatIsBelowParallel(
                keypoints
            ):
                self.squatComingBackUp = True
            elif self.squatWasAtTopLastFrame and not squatIsAtTheTop(keypoints):
                self.squatComingDown = True
            elif self.squatComingBackUp and squatIsAtTheTop(keypoints):
                self.squatRep += 1
                self.squatComingBackUp = False

            self.squatWasBelowParallelLastFrame = squatIsBelowParallel(keypoints)
            self.squatWasAtTopLastFrame = squatIsAtTheTop(keypoints)

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
