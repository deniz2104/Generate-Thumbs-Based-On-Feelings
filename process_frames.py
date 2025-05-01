from decord import VideoReader
from decord import cpu
import os
import cv2
import concurrent.futures

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