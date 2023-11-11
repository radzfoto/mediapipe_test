import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
from mediapipe.tasks.python import vision

def get_hand_connections_as_tuples(connections: list) -> list[tuple[int, int]]:
    # Convert each Connection object into a tuple and add to the list
    connections_tuples: list[tuple[int, int]] = [(conn.start, conn.end) for conn in connections]
    return connections_tuples

# Initialize MediaPipe Hands.
hand_connections: list = vision.HandLandmarksConnections.HAND_CONNECTIONS
hand_connections: list = get_hand_connections_as_tuples(hand_connections)

def configure_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap # configure_camera()

def capture(cap: cv2.VideoCapture) -> np.ndarray | None:
    success, frame = cap.read()
    frame_rgb = None
    if success:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb # capture()

def process_frame(frame: np.ndarray):
    with Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
        results = hands.process(frame)
        return results # process_frame

def draw_landmarks(frame: np.ndarray, landmarks):
    # Draw the hand landmarks in place in the frame
    if landmarks.multi_hand_landmarks:
        for hand_landmarks in landmarks.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame, hand_landmarks, hand_connections)
    return # draw_landmarks()

def display_image(frame):
    # Display the image (modify this part based on your environment)
    cv2.imshow('Processed Image', frame)
    return # display_image()

def main():
    camera_id: int = 2
    width, height = (1920, 1080) if camera_id == 2 else (1280, 720)

    cap = configure_camera(camera_id, width, height)

    while cap.isOpened():
        frame: np.ndarray | None = capture(cap)
        if frame is not None:
            landmarks = process_frame(frame)
            draw_landmarks(frame, landmarks)
 
        display_image(frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
    # end while
    
    return # main()

if __name__ == "__main__":
    main()