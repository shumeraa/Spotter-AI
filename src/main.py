import cv2
from squatAnalyzer import SquatAnalyzer

if __name__ == "__main__":
    recordingsFolder = "Recordings"
    videoPath = r"Data\Half Rep Squat.mp4"  # 0 for webcam
    analyzer = SquatAnalyzer(recordingsFolder)

    try:
        analyzer.emptyRecordingsFolder()

        # Open the video stream from a file
        cap = cv2.VideoCapture(videoPath)
        cv2.namedWindow("Squat Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Squat Video", 1280, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, _ = analyzer.process_frame(frame)
            cv2.imshow("Example", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("Stopping analysis")
        analyzer.cleanup()

    except Exception as e:
        print(f"An error occurred: {e}")
