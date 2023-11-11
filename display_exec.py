import asyncio
import base64
import cv2
import json
import numpy as np
import websockets

async def receive_and_display():
    uri = "ws://localhost:1880/ws/detect"  # WebSocket URI for receiving processed frames
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            image_data = base64.b64decode(data['image'])
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('Processed Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

asyncio.get_event_loop().run_until_complete(receive_and_display())
cv2.destroyAllWindows()
