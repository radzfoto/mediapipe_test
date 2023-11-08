from deepface import DeepFace
import cv2

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 128)  # red

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Unable to capture video")
        break

    # Apply face detection
    detections = DeepFace.functions.extract_faces(frame, detector_backend = 'retinaface', enforce_detection=False)
    if detections is None:
        detections = []

    # Draw bounding boxes around detected faces
    for detection in detections:
        face = detection[1]
        confidence = detection[2]
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text_location = (MARGIN + x,
                     MARGIN + ROW_SIZE + y)
        cv2.putText(frame, str(confidence)[:4], text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Display the resulting frame
    cv2.imshow('Webcam - Face Detection', frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      cap.release()
      break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
