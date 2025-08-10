import cv2
import numpy as np
from keras.models import load_model

model = load_model('model_file_30epochs.h5')

video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()





    
    
    
    
    
# import cv2
# import numpy as np
# from keras.models import load_model
# import os
# import random
# from playsound import playsound

# # Load the trained model
# model = load_model('model_file_30epochs.h5')

# # Load Haar Cascade for face detection
# faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Emotion labels
# labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# # Define music for each emotion
# music_dict = {
#     'Angry': ['music/angry1.mp3', 'music/angry2.mp3'],
#     'Disgust': ['music/disgust1.mp3', 'music/disgust2.mp3'],
#     'Fear': ['music/fear1.mp3', 'music/fear2.mp3'],
#     'Happy': ['music/happy1.mp3', 'music/happy2.mp3'],
#     'Neutral': ['music/neutral1.mp3', 'music/neutral2.mp3'],
#     'Sad': ['music/sad1.mp3', 'music/sad2.mp3'],
#     'Surprise': ['music/surprise1.mp3', 'music/surprise2.mp3']
# }

# # Function to play a song based on detected emotion
# def play_music(emotion):
#     if emotion in music_dict:
#         song = random.choice(music_dict[emotion])  # Pick a random song for variety
#         if os.path.exists(song):
#             playsound(song)
#         else:
#             print(f"⚠️ Music file {song} not found!")

# # Start video capture
# video = cv2.VideoCapture(0)

# if not video.isOpened():
#     print("❌ Failed to open camera.")
#     exit()

# print("✅ Camera is ready. Press 'q' to quit.")

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("⚠️ Failed to read frame.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

#     for x, y, w, h in faces:
#         sub_face_img = gray[y:y+h, x:x+w]
#         resized = cv2.resize(sub_face_img, (48, 48))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 48, 48, 1))
        
#         result = model.predict(reshaped)
#         label = np.argmax(result, axis=1)[0]
#         emotion = labels_dict[label]

#         # Draw bounding box & emotion label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         # Play music based on detected emotion
#         play_music(emotion)

#     cv2.imshow("Emotion Detection", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting...")
#         break

# video.release()
# cv2.destroyAllWindows()
