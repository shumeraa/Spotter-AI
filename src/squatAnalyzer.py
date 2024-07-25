from ultralytics import YOLO
import cv2
from analyzeSquat import (
    getKneeAngle,
    plotKneeAngle,
    squatIsBelowParallel,
    plotRepCount,
    squatIsAtTheTop,
)
import multiprocessing
from callLLM import callLLMs
import os

missedDepthString = "The client missed depth on the squat"


class SquatAnalyzer:
    def __init__(self, model, recordingsFolder, ):
       