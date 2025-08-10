from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# Class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Optional: Display one batch (remove if not needed)
# img, label = next(train_generator)

# Define model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Count the number of training and testing images
def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([f for f in files if f.endswith('.jpg') or f.endswith('.png')])  # Count only image files
    return count

num_train_imgs = count_images(train_data_dir)
num_test_imgs = count_images(validation_data_dir)

print("Number of training images:", num_train_imgs)
print("Number of testing images:", num_test_imgs)

epochs = 30

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,  # Use integer division for steps
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32  # Use integer division for validation steps
)

# Save the trained model
# model.save('model_file.h5')
model.save('model_file_30epochs.h5')  # Add this line at the end of your training script







# import cv2
# import numpy as np
# import pygame  # For playing music
# import os
# from tensorflow.keras.models import load_model

# # Load pre-trained model
# model = load_model("model_file_30epochs.h5")

# # Emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Emotion-to-music mapping (Replace with your music file paths)
# emotion_music = {
#     'Angry': 'music/rock.mp3',
#     'Disgust': 'music/melancholy.mp3',
#     'Fear': 'music/calm.mp3',
#     'Happy': 'music/upbeat.mp3',
#     'Neutral': 'music/neutral.mp3',
#     'Sad': 'music/sad.mp3',
#     'Surprise': 'music/surprise.mp3'
# }

# # Initialize pygame for music playback
# pygame.mixer.init()

# # Load OpenCV face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Open webcam
# cap = cv2.VideoCapture(0)

# current_song = None  # Track currently playing song

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for x, y, w, h in faces:
#         face_roi = gray[y:y+h, x:x+w]  # Extract face region
#         resized = cv2.resize(face_roi, (48, 48)) / 255.0  # Normalize
#         reshaped = np.reshape(resized, (1, 48, 48, 1))  # Reshape for model
#         result = model.predict(reshaped)
#         label = np.argmax(result)  # Get predicted emotion

#         # Draw bounding box & label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, emotion_labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         # Play music based on detected emotion
#         song_path = emotion_music[emotion_labels[label]]
#         if current_song != song_path:  # Avoid restarting the same song
#             pygame.mixer.music.load(song_path)
#             pygame.mixer.music.play()
#             current_song = song_path

#     cv2.imshow("Emotion Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()
