import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
try:
    model = load_model('model_file.h5')
except Exception as e:
    print("Error loading model:", e)
    exit()

# Load the Haar Cascade classifier for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceDetect.empty():
    print("Error: Haar Cascade XML file not loaded!")
    exit()

# Define emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the image
frame = cv2.imread("data.jpg")
if frame is None:
    print("Error: Image not found!")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

if len(faces) == 0:
    print("No faces detected!")
else:
    # Process each detected face
    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict emotion
        try:
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
        except Exception as e:
            print("Error during prediction:", e)
            continue

        # Draw rectangles and labels on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display the processed image
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
