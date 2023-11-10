import asyncio
import base64
import cv2
import json
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands
import mediapipe.python.solutions as mp_solutions
import numpy as np
import websockets

async def detect_and_send():
    uri_receive = "ws://localhost:1880/ws/capture"  # WebSocket URI for receiving frames
    uri_send = "ws://localhost:1880/ws/detect"      # WebSocket URI for sending processed frames
    
    async with websockets.connect(uri_receive) as websocket_receive, \
               websockets.connect(uri_send) as websocket_send:
        with Hands(static_image_mode=False,
                                  max_num_hands=2,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as hands:

            async for message in websocket_receive:
                # Process the frame
                data = json.loads(message)
                image_data = base64.b64decode(data['image'])
                frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

                # Perform hand detection
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                annotated_image = frame.copy()

                # Draw hand landmarks
                if results is not None and hasattr(results, 'multi_hand_landmarks'):
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_solutions.drawing_utils.draw_landmarks(annotated_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Encode frame to JPEG format and then base64 encode
                ret, buffer = cv2.imencode('.jpg', annotated_image)
                if ret:
                    jpg_as_bytes = buffer.tobytes()
                    jpg_as_text = base64.b64encode(jpg_as_bytes).decode('utf-8')
                    
                    # Send the base64 encoded image
                    await websocket_send.send(json.dumps({"image": jpg_as_text}))

asyncio.get_event_loop().run_until_complete(detect_and_send())
