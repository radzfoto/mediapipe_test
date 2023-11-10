import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
from mediapipe.tasks.python import vision

def get_hand_connections_as_tuples(connections: list) -> list[tuple[int, int]]:
    # Convert each Connection object into a tuple and add to the list
    connections_tuples: list[tuple[int, int]] = [(conn.start, conn.end) for conn in connections]
    return connections_tuples

def main():
    # Initialize MediaPipe Hands.
    hand_connections: list = vision.HandLandmarksConnections.HAND_CONNECTIONS
    hand_connections: list = get_hand_connections_as_tuples(hand_connections)

    with Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:

        cap = cv2.VideoCapture(0)  # Start video capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                exit(1)
            # frame = cv2.imread('/home/raul/Downloads/hands.jpg')
            # frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Draw the hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(
                        frame, hand_landmarks, hand_connections)

            # Display the annotated frame
            cv2.imshow('Hand Tracking', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()