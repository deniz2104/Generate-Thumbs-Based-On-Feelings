import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.chdir(project_root)

from src.utils.process_frames import *
from src.utils.prepare_yolo import *
from src.detection.face_detection_yolov8 import detect_faces_yolo, train_yolo_model
from src.detection.face_detection_mediapipe import detect_faces
from src.detection.filter_motion import eliminate_photos_with_motion
from src.detection.filter_closed_eyes import eliminate_closed_eyes
from src.utils.get_main_person import detect_most_present_face
from src.predict_emotions import classify_faces_in_directory

if __name__ == '__main__':
    video_to_frames_one_per_second(video_path='The Present.mp4', frames_dir='data/test_frames', overwrite=True)
    detect_faces(image_directory='data/test_frames', save_directory='output/test_frames_faces')
    weight_path = train_yolo_model(data_yaml='config/widerface_yolo.yaml', img_size=320, batch_size=16, epochs=5)
    detect_faces_yolo(weight_path="data/widerface_yolo/yolo_model/weights/best.pt", image_directory='output/test_frames_faces', save_directory='output/test_frames_faces_yolo', conf_thresh=0.6)
    eliminate_closed_eyes(
        image_directory='output/test_frames_faces_yolo',
        save_directory='output/test_frames_without_closed_eyes',
        ear_threshold=0.21
    )
    eliminate_photos_with_motion(image_directory='output/test_frames_without_closed_eyes', fft_threshold=140, radius=60, save_directory='output/test_frames_without_motion')
    detect_most_present_face(image_directory='output/test_frames_without_motion', save_directory='output/final_frames')
    classify_faces_in_directory(image_directory='output/final_frames', save_dir='data/emotions')
    