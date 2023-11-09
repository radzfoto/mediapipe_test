# This script would be saved as display.py
import cv2
import sys
import base64
import json
import numpy as np

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

# Display the resulting image
cv2.imshow('Processed Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
