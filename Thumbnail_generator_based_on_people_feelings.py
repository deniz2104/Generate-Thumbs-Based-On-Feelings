import cv2
import numpy as np
import mediapipe as mp
from decord import VideoReader
from decord import cpu
from PIL import Image
import os
import concurrent.futures
from ultralytics import YOLO



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
def detect_face_yolov8(image_path, model, confidence_threshold=0.25):
    results = model(image_path)
    for box in results[0].boxes:
        if box.conf[0] >= confidence_threshold:
            return True
    return False

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
def save_frame(frame, save_path, overwrite):
    if not os.path.exists(save_path) or overwrite:
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return 1
    return 0

def extract_frames_one_per_second(video_path, frames_dir, overwrite=False):
    
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    video_dir, video_filename = os.path.split(video_path)
    
    assert os.path.exists(video_path)
    
    vr = VideoReader(video_path, ctx=cpu(0))
    
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    os.makedirs(os.path.join(frames_dir), exist_ok=True)
    
    saved_count = 0
    frame_indices = []
    
    for sec in range(int(total_frames / fps) + 1):
        frame_idx = int(sec * fps)
        if frame_idx < total_frames:
            frame_indices.append(frame_idx)
    
    frames = vr.get_batch(frame_indices).asnumpy()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx, frame in enumerate(frames):
            save_path = os.path.join(frames_dir, f"second_{idx:04d}.jpg")
            futures.append(executor.submit(save_frame, frame, save_path, overwrite))
        for future in concurrent.futures.as_completed(futures):
            saved_count += future.result()

    print(f"Extracted {saved_count} frames at 1-second intervals")
    return saved_count


def video_to_frames_one_per_second(video_path, frames_dir, overwrite=False):
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    video_dir, video_filename = os.path.split(video_path)
    
    os.makedirs(os.path.join(frames_dir), exist_ok=True)
    
    print(f"Extracting one frame per second from {video_filename}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(extract_frames_one_per_second, video_path, frames_dir, overwrite)
        future.result()    
    return os.path.join(frames_dir, video_filename)


if __name__ == '__main__':
    #video_to_frames_one_per_second(video_path='The Present.mp4', frames_dir='test_frames', overwrite=True)
    detect_faces(image_directory='test_frames',save_directory='test_frames_faces')