import cv2
import numpy as np
def get_face_boxxes(image_path, model, conf_thresh=0.7):
    image = cv2.imread(image_path)
    results = model.predict(image, conf=conf_thresh,verbose=False)[0]
    bboxes = results.boxes.xyxy.cpu().numpy().astype(int)
    return bboxes

def crop_face(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def is_motion_detected_using_fft(face_roi, fft_threshold=140, radius=60):
    face_resized = cv2.resize(face_roi, (256, 256))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-10)
    
    h, w = gray.shape
    cy, cx = h//2, w//2
    fft_roi = magnitude_spectrum[cy - radius:cy + radius, cx - radius:cx + radius]
    mean_magnitude = np.mean(fft_roi)
    print(f"Mean FFT Magnitude: {mean_magnitude}")
    return mean_magnitude < fft_threshold


def is_image_blurred(image_path, model, fft_threshold=140, radius=60):
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    bboxes = get_face_boxxes(image_path, model)
    if len(bboxes) == 0:
        return False
    
    for bbox in bboxes:
        face_roi = crop_face(image, bbox)
        if is_motion_detected_using_fft(face_roi, fft_threshold, radius):
            return True
    return False