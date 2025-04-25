import cv2
import numpy as np
import mediapipe as mp
import torch
from decord import VideoReader
from decord import cpu
from PIL import Image
import os
import concurrent.futures
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf

torch.cuda.empty_cache()
mp_face_detection = mp.solutions.face_detection



def prepare_dataset():
    image_dir = "images/"
    label_dir = "labels/"
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    assert len(images) == len(labels), f"Mismatch: {len(images)} images and {len(labels)} labels!"

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    for img, lbl in zip(train_images, train_labels):
        shutil.copy(os.path.join(image_dir, img), "dataset/images/train")
        shutil.copy(os.path.join(label_dir, lbl), "dataset/labels/train")

    for img, lbl in zip(val_images, val_labels):
        shutil.copy(os.path.join(image_dir, img), "dataset/images/val")
        shutil.copy(os.path.join(label_dir, lbl), "dataset/labels/val")

    print("Dataset has been split into train and validation sets.")

    data_yaml = """
    train: dataset/images/train
    val: dataset/images/val
    nc: 1
    names: ['face']
    """

    with open("widerface_yolo.yaml", "w") as f:
        f.write(data_yaml)

    print("Data configuration file 'widerface_yolo.yaml' created.")
    return "widerface_yolo.yaml"

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

def get_device():
    if torch.cuda.is_available():
        return "0"  
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  
    else:
        return "cpu"

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

def train_yolo_model(data_yaml,img_size=420,batch_size=16,epochs=20):
    device = get_device()
    os.makedirs("widerface_yolo",exist_ok=True)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        epochs=epochs,
        device=device,
        project="widerface_yolo",
        workers=0,
        name="yolo_model",
        exist_ok=True
    )
    weights_path = os.path.join("widerface_yolo", "yolo_model", "weights", "best.pt")
    return weights_path    
def save_frame(frame, save_path, overwrite):
    if not os.path.exists(save_path) or overwrite:
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return 1
    return 0

def detect_faces_yolo(weight_path, image_directory, save_directory=None, conf_thresh=0.7):
    model = YOLO(weight_path)
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image in image_files:
        image_path = os.path.join(image_directory, image)
        results = model(image_path, conf=conf_thresh)[0]
        if len(results.boxes) > 0:
            img = cv2.imread(image_path)
            save_path = os.path.join(save_directory, image)
            cv2.imwrite(save_path, img)

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
    #weight_path = train_yolo_model(data_yaml='widerface_yolo.yaml', img_size=320, batch_size=16, epochs=5)
    detect_faces_yolo(weight_path="widerface_yolo/yolo_model/weights/best.pt", image_directory='test_frames_faces', save_directory='test_frames_faces_yolo',conf_thresh=0.6)