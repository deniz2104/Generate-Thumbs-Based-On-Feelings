import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os
from detect_faces_having_motion_with_fft import get_face_boxxes, crop_face

my_model=load_model("fer_model.h5",compile=False)
width=48
height=48
labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def setup_emotion_directories(base_dir='emotions'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for emotion in labels:
        emotion_dir = os.path.join(base_dir, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

def classify_face_emotion(image_path, model, save_dir='emotions'):
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    bboxes = get_face_boxxes(image_path, model)
    if len(bboxes) == 0:
        return False
    
    emotions_detected = []
    for bbox in bboxes:
        face_roi = crop_face(image, bbox)
        face_roi = cv2.resize(face_roi, (width, height))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = face_roi.reshape(1, width, height, 1)
        
        predictions = my_model.predict(face_roi)
        emotion_index = predictions.argmax()
        emotion_label = labels[emotion_index]
        emotions_detected.append(emotion_label)
        print(f"Detected emotion: {emotion_label}")
    
    return emotions_detected[0] if emotions_detected else None

def classify_faces_in_directory(image_directory, save_dir='emotions'):
    setup_emotion_directories(save_dir)
    
    model = YOLO("widerface_yolo/yolo_model/weights/best.pt", verbose=False)
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image in image_files:
        image_path = os.path.join(image_directory, image)
        print(f"\nProcessing: {image_path}")
        
        emotion = classify_face_emotion(image_path, model)
        if emotion:
            save_path = os.path.join(save_dir, emotion, image)
            img = cv2.imread(image_path)
            cv2.imwrite(save_path, img)
            print(f"Saved to {emotion} folder: {image}")