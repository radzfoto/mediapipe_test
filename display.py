import base64
import cv2
import numpy as np

def display_image(input_data):
    if input_data is None:
        height: int = 1080
        width: int = 1920
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        # Decode the image
        img_data = base64.b64decode(input_data)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Display the image (modify this part based on your environment)
    cv2.imshow('Processed Image', frame)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return {'payload': 'Image displayed'}

msg = display_image(msg['payload'])
