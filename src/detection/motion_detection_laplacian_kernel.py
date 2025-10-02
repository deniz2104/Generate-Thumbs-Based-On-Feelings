import cv2
import numpy as np
def is_motion_detected_using_laplacian_with_kernel(image_path,threshold=80):
    image=cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel=np.array([[-1,2,-1]],dtype=np.float32)
    dx = cv2.filter2D(gray, cv2.CV_64F, kernel)
    dy = cv2.filter2D(gray, cv2.CV_64F, kernel.T)
    modified_lap = np.abs(dx) + np.abs(dy)
    laplacian_var = np.var(modified_lap)
    return laplacian_var < threshold
