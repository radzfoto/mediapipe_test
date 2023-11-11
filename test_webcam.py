import sys
import cv2

def test_webcam():
    # Open a video stream from the default webcam (usually webcam 0)
      # Start capturing video input from the camera
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )
        # If frame is read correctly, display it
        frame = cv2.flip(frame, 1)
        cv2.imshow('Webcam Test', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()
