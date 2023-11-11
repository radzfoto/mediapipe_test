import base64
import cv2
import json
import numpy as np
from mediapipe.python.solutions.hands import Hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
from mediapipe.tasks.python import vision

def get_hand_connections_as_tuples(connections):
    return [(conn.start, conn.end) for conn in connections]

def process_frame(input_data):
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    hand_connections_tuples = get_hand_connections_as_tuples(hand_connections)

    # Decode the image
    image_data = base64.b64decode(input_data)
    frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Perform hand detection
    with Hands(static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_image = frame.copy()

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(annotated_image, hand_landmarks, hand_connections_tuples)

    # Encode frame to JPEG format and then base64 encode
    ret, buffer = cv2.imencode('.jpg', annotated_image)
    if ret:
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return {'payload': jpg_as_text}

# Example usage
msg = {'payload': '<your_base64_encoded_image_string>'}
msg = process_frame(msg['payload'])
