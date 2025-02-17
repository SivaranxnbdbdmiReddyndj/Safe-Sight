from fer import FER
import cv2

# Initialize the FER model
detector = FER()

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the frame for emotions
    result = detector.detect_emotions(frame)
    for face in result:
        (x, y, w, h) = face["box"]
        emotion = max(face["emotions"], key=face["emotions"].get)

        # Draw rectangle and label emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
