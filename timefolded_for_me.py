import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_detection_model = load_model('emotion_detection_model.h5')
emotion_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
##### !!!
# cap = cv2.VideoCapture('2_video_test.mp4')

###### Capture img from video and save in folder ######
absolute_path = "C:/Users/PC/Desktop/AI_Techs_class/test_python3.9"
num_vid = 3
# Load the video file - absolute path
video_path = f"C:/Users/PC/Desktop/AI_Techs_class/{num_vid}_video_test.mp4"
cap = cv2.VideoCapture(video_path)

# Create a folder to save the images - absolute path
fold_cap = f"C:/Users/PC/Desktop/AI_Techs_class/test_python3.9/img_from_vid{num_vid}"
if not os.path.exists(fold_cap):
    os.makedirs(fold_cap)

# Set the time interval in seconds
interval = 5
frame_rate = cap.get(cv2.CAP_PROP_FPS)
# Calculate the frame interval
frame_interval = int(frame_rate * interval)
# Initialize frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    # Break the loop when no more frames are read
    if not ret:
        break
    # Save the frame every - seconds
    if frame_count % interval == 0:
        image_path = f"img_from_vid{num_vid}/frame_{frame_count}.jpg"
        cv2.imwrite(image_path, frame)
    # Increment the frame count
    frame_count += 1

cap.release()
cv2.waitKey(0)

######## face detection from folder #########
image_folder = f'img_from_vid{num_vid}'
# start_time = 0.0
#os.listdir(image_folder)
# video_path
for filename in os.listdir(image_folder):
    # ret, frame = cap.read()
    frame = cv2.imread(os.path.join(image_folder, filename))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection_model.detectMultiScale(
        gray,
        scaleFactor = 1.3, #1.3
        minNeighbors = 8, #6
        minSize = (140, 140)
        ) #, nope, minSize=(25, 25) | minSize = (150, 150)

    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.reshape(face_roi, [1, face_roi.shape[0], face_roi.shape[1], 1])
        face_roi = face_roi / 255.0

        emotion_label = np.argmax(emotion_detection_model.predict(face_roi), axis=1)[0]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,130,0), 2)
        cv2.putText(frame, emotion_labels[emotion_label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,165,0), 2)
    cv2.imshow('Face Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()