import asyncio
import cv2
import base64
import websockets
import json

async def capture_and_send():
    camera_id: int = 2

    width: int = 1280
    height: int = 720
    if camera_id == 0:
        width: int = 1280
        height: int = 720
    elif camera_id == 2:
        width: int = 1920
        height: int = 1080

    cap = cv2.VideoCapture(camera_id)  # Start video capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    uri = "ws://localhost:1880/ws/capture"  # WebSocket URI
    async with websockets.connect(uri) as websocket:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Encode frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    # Convert numpy array to bytes and then base64 encode
                    jpg_as_bytes = buffer.tobytes()
                    jpg_as_text = base64.b64encode(jpg_as_bytes).decode('utf-8')
                    
                    # Send the base64 encoded image
                    await websocket.send(json.dumps({"image": jpg_as_text}))
                await asyncio.sleep(0.1)  # Adjust frame rate

asyncio.get_event_loop().run_until_complete(capture_and_send())
