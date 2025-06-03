import os
from ultralytics import YOLO
from PIL import Image
import cv2
from detect_faces_having_motion_with_fft import *
from detect_faces_having_motion_with_laplacian import *
from detect_faces_having_motion_with_laplacian_with_kernel import *
def eliminate_photos_with_motion(image_directory, fft_threshold=140, radius=60, save_directory=None):
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model = YOLO("widerface_yolo/yolo_model/weights/best.pt", verbose=False)
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image in image_files:
        image_path = os.path.join(image_directory, image)
        img= cv2.imread(image_path)
        if not (is_motion_detected_using_laplacian(image_path, threshold=80) or
            is_motion_detected_using_laplacian_with_kernel(image_path, threshold=80) or
            is_image_blurred(image_path, model, fft_threshold=fft_threshold, radius=radius)):
            if save_directory:
                img = Image.open(image_path)
                img.save(os.path.join(save_directory, image))