import mediapipe as mp
import cv2
import concurrent.futures
import os
from PIL import Image
mp_face_detection = mp.solutions.face_detection

def detect_faces_mediapipe(image_path, min_detection_confidence=0.7):
    image = cv2.imread(image_path)
    if image is None:
        return False
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_detection_confidence
    ) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.detections is not None and len(results.detections) > 0
def detect_faces(image_directory, save_directory=None):
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
    image_files = os.listdir(image_directory)

    def process_image(image_file):
        image_path = os.path.join(image_directory, image_file)
        try:
            if not detect_faces_mediapipe(image_path):
                return
            if save_directory:
                img = Image.open(image_path)
                img.save(os.path.join(save_directory, image_file))  
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, image_files)