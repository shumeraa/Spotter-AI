import cv2
from squatAnalyzer import SquatAnalyzer

if __name__ == "__main__":
    recordingsFolder = "Recordings"
    videoPath = r"Data\shortSquat1.mp4"  # 0 for webcam
    analyzer = SquatAnalyzer(recordingsFolder)

    try:
        analyzer.emptyRecordingsFolder()

        # Open the video stream from a file
        cap = cv2.VideoCapture(videoPath)
        cv2.namedWindow("Squat Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Squat Video", 1280, 720)

        # for recording the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        outputPath = r"Data\Recordings\shortSquatPose.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, _ = analyzer.process_frame(frame)
            cv2.imshow("Squat Video", annotated_frame)

            out.write(annotated_frame)  # for recording

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()  # for recording
        cv2.destroyAllWindows()

        print("Stopping analysis")
        analyzer.cleanup()

    except Exception as e:
        print(f"An error occurred: {e}")
