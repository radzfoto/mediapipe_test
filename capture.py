import cv2
import base64

def capture_frame():
    camera_id = 2
    width, height = (1920, 1080) if camera_id == 2 else (1280, 720)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = cap.read()
    cap.release()

    if ret:
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            # Convert numpy array to bytes and then base64 encode
            jpg_as_bytes = buffer.tobytes()
            jpg_as_text = base64.b64encode(jpg_as_bytes).decode('utf-8')
            return {'payload': jpg_as_text}

    return {'payload': None}

msg = capture_frame()
