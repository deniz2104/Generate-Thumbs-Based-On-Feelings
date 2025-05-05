import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os
from detect_faces_having_motion_with_fft import get_face_boxxes, crop_face

my_model=load_model("fer_model.h5",compile=False)
width=48
height=48
labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def classify_face_emotion(image_path,model):
    image=cv2.imread(image_path)
    if image is None:
        return False
    bboxes = get_face_boxxes(image_path, model)
    if len(bboxes) == 0:
        return False
    
    for bbox in bboxes:
        face_roi = crop_face(image, bbox)
        face_roi = cv2.resize(face_roi, (width, height))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = face_roi.reshape(1, width, height, 1)
        
        predictions = my_model.predict(face_roi)
        emotion_index = predictions.argmax()
        emotion_label = labels[emotion_index]
        print(f"Detected emotion: {emotion_label}")

def classify_faces_in_directory(image_directory):
    model=YOLO("widerface_yolo/yolo_model/weights/best.pt",verbose=False)
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image in image_files:
        image_path = os.path.join(image_directory, image)
        print(f"\nProcessing: {image_path}")
        classify_face_emotion(image_path,model)