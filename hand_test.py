import cv2
from mediapipe.python.solutions.hands import Hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
from mediapipe.tasks.python.vision import HandLandmarksConnections

# Setup and configure webcam
cap = cv2.VideoCapture(0)

# Initialize mediapipe hand solution
hands = Hands()

def get_hand_connections_as_tuples(connections: list) -> list[tuple[int, int]]:
    # Convert each Connection object into a tuple and add to the list
    connections_tuples: list[tuple[int, int]] = [(conn.start, conn.end) for conn in connections]
    return connections_tuples

# Initialize MediaPipe Hands connections
hand_connections: list = HandLandmarksConnections.HAND_CONNECTIONS
hand_connections: list = get_hand_connections_as_tuples(hand_connections)

def detect_hand_landmarks(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame
    results = hands.process(rgb_frame)
    # Return the hand landmark data
    return results.multi_hand_landmarks

def draw_landmarks(frame, landmarks):
    if landmarks:
        for hand_landmarks in landmarks:
            drawing_utils.draw_landmarks(
                frame, hand_landmarks, hand_connections)

def display_frame(frame):
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = detect_hand_landmarks(frame)
    draw_landmarks(frame, landmarks)

    if display_frame(frame):
        break

cap.release()
cv2.destroyAllWindows()
