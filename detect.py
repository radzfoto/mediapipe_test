# This script would be saved as detect.py
import cv2
import mediapipe as mp
import sys
import base64
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Receive base64 encoded image from stdin
input_data = json.loads(sys.stdin.read())
image_data = input_data.get("image")

if not image_data:
    print(json.dumps({"error": "No image data"}))
    sys.exit(1)

# Convert base64 to image
jpg_original = base64.b64decode(image_data)
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
frame = cv2.imdecode(jpg_as_np, flags=1)

# Process the image and find hands
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Draw the hand annotations on the image.
annotated_image = frame.copy()
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Convert the annotated image to JPEG and then to a base64 string
retval, buffer = cv2.imencode('.jpg', annotated_image)
jpg_as_text = base64.b64encode(buffer).decode('utf-8')

print(json.dumps({"image": jpg_as_text}))

hands.close()
