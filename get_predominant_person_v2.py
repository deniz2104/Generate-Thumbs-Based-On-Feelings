import mediapipe as mp
import cv2
import os
import numpy as np
mp_face_mesh = mp.solutions.face_mesh

def extract_face_landmarks(image_path):
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
            return None
        landmarks=results.multi_face_landmarks[0].landmark
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

def calculate_landmark_distance(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return float('inf')
    if len(landmarks1) != len(landmarks2):
        return float('inf')
    distance = np.mean(np.linalg.norm(landmarks1 - landmarks2,axis=1))
    return distance

def detect_most_present_face(image_directory):
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return None
    first_landmark = extract_face_landmarks(os.path.join(image_directory, image_files[0]))
    for image in image_files:
        image_path = os.path.join(image_directory, image)
        landmarks = extract_face_landmarks(image_path)
        distance = calculate_landmark_distance(first_landmark, landmarks)
        ## e ok,iau varianta cu infinit si aia e tot ce mi trebuie,maine asamblez totul si trec la recunostoarea de emotii