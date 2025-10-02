import mediapipe as mp
import cv2
import numpy as np
import os
from PIL import Image

mp_face_mesh = mp.solutions.face_mesh
FACE_MESH_EYES = [
    [33, 160, 158, 133, 153, 144],
    [362, 385, 387, 263, 373, 380]
]

def calculate_ear(eye_points):
    d1 = np.linalg.norm(eye_points[1] - eye_points[5])
    d2 = np.linalg.norm(eye_points[2] - eye_points[4])
    
    d3 = np.linalg.norm(eye_points[0] - eye_points[3])
    
    return (d1 + d2) / (2 * d3)

def detect_closed_eyes(image_path, ear_threshold=0.21):
    image = cv2.imread(image_path)
    if image is None:
        return False
        
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return True 

        for face_landmarks in results.multi_face_landmarks:
            for eye_indices in FACE_MESH_EYES:
                eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
                                    face_landmarks.landmark[i].y * image.shape[0])
                                    for i in eye_indices])
                ear = calculate_ear(eye_points)
            if ear < ear_threshold and ear:
                return True
        return False

def eliminate_closed_eyes(image_directory, save_directory=None, ear_threshold=0.21):
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    image_files = [f for f in os.listdir(image_directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        img=cv2.imread(image_path)
        print(image_path)
        if not detect_closed_eyes(image_path, ear_threshold=ear_threshold):
            if save_directory:
                img = Image.open(image_path)
                img.save(os.path.join(save_directory, image_file))
