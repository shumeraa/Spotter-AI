from ultralytics import YOLO
import monitorAudio
import cv2
from analyzeSquat import (
    getKneeAngle,
    plotKneeAngle,
    squatIsBelowParallel,
    plotRepCount,
    squatIsAtTheTop,
)
import multiprocessing
from callLLM import callLLM


def llmCall_worker(input_queue, output_queue):
    while True:
        input_string = input_queue.get()
        if input_string is None:  # Sentinel value to signal the end of the process
            break
        callLLM(input_string)
        output_queue.put(f"Processed: {input_string}")


if __name__ == "__main__":
    try:
        # load a pretrained YOLOv8m model
        model = YOLO("yolov8m-pose.pt")

        path_to_monitor = r"Recordings"
        monitoring_process = multiprocessing.Process(
            target=monitorAudio.start_monitoring, args=(path_to_monitor,)
        )
        monitoring_process.start()

        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        llmCall_process = multiprocessing.Process(
            target=llmCall_worker, args=(input_queue, output_queue)
        )
        llmCall_process.start()

        # Open the video stream from a file
        cap = cv2.VideoCapture(r"Data\Squat.mp4")
        # cap = cv2.VideoCapture(0)
        cv2.namedWindow("Example", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Example", 1280, 720)
        squatWasBelowParallel = False
        onFirstFrame = True
        count = 0
        playVideo = 1
        squatRep = 0
        squatComingBackUp = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use a model to detect keypoints in the frame and convert them to numpy array
            results = model(frame)
            annotated_frame = results[0].plot()
            keypoints = results[0].keypoints.xyn.cpu().numpy()

            left_knee_angle = getKneeAngle(annotated_frame, keypoints, left=True)
            right_knee_angle = getKneeAngle(annotated_frame, keypoints, left=False)

            if left_knee_angle is None or right_knee_angle is None:
                count += 1
            else:
                if onFirstFrame:
                    # get the initial state of the squat
                    onFirstFrame = False
                    squatWasBelowParallel = squatIsBelowParallel(keypoints)
                if not onFirstFrame:
                    # check if the squat has successfully gone below parallel and is coming back up
                    if (
                        squatWasBelowParallel is True
                        and squatIsBelowParallel(keypoints) is False
                    ):
                        squatComingBackUp = True
                    elif squatComingBackUp and squatIsAtTheTop(keypoints) is True:
                        squatRep += 1
                        input_queue.put(squatRep)
                        squatComingBackUp = False

                    squatWasBelowParallel = squatIsBelowParallel(keypoints)

                plotKneeAngle(annotated_frame, keypoints, left_knee_angle, left=True)
                plotKneeAngle(annotated_frame, keypoints, right_knee_angle, left=False)

            plotRepCount(annotated_frame, squatRep)

            # if (
            #     left_knee_angle
            #     and right_knee_angle
            #     and left_knee_angle > 160
            #     and right_knee_angle > 160
            # ):
            #     # pause the video if the person is in the lower portion of the squat
            #     playVideo = 0

            # Display the frame with the annotated angle in a window
            cv2.imshow("Example", annotated_frame)

            # Introduce a delay between frames and break the loop if 'q' is pressed
            if cv2.waitKey(playVideo) & 0xFF == ord("q"):
                break

        # Release the video capture object and close all OpenCV windows
        print(f"Number of frames with missing keypoints: {count}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        input_queue.put(None)
        llmCall_process.join()
        monitorAudio.stop_monitoring()
        monitoring_process.join()
        cap.release()
        cv2.destroyAllWindows()
