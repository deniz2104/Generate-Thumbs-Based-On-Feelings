import cv2
def is_motion_detected_using_laplacian(image_path,threshold=80):
    image=cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    return laplacian_var < threshold