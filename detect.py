import asyncio
import base64
import cv2
import json
from mediapipe.python.solutions.hands import Hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
from mediapipe.tasks.python import vision
import numpy as np
import websockets

import base64
import cv2
import numpy as np

def process_frame(input_data):
    if input_data is None:
        return {'payload': None}

    # Placeholder for your detection logic
    # Decode the image
    img_data = base64.b64decode(input_data)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Process frame (Placeholder logic)
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Encode back to base64
    ret, buffer = cv2.imencode('.jpg', processed_frame)
    jpg_as_bytes = buffer.tobytes()
    jpg_as_text = base64.b64encode(jpg_as_bytes).decode('utf-8')

    return {'payload': jpg_as_text}

msg = process_frame(msg['payload'])
