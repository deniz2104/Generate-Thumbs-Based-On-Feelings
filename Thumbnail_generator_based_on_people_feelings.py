import cv2
import numpy as np
from decord import VideoReader
from decord import cpu,gpu
import face_alignment
import torch
from PIL import Image
import os
import concurrent.futures

## TODO: need to find a robust way to eliminate the frames that are not needed from the video

def detect_faces(image_directory, save_directory=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
    image_files = os.listdir(image_directory)
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        try:
            landmarks = fa.get_landmarks(image_path)
            if landmarks is None or len(landmarks) == 0:
                print(f"No faces detected in {image_path}")
                continue
            if save_directory:
                img = Image.open(image_path)
                img.save(os.path.join(save_directory, image_file))  
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue   
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
    """
    Extracts one frame per second from a video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :return: path to the directory where the frames were saved
    """
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
    video_to_frames_one_per_second(video_path='The Present.mp4', frames_dir='test_frames', overwrite=True)