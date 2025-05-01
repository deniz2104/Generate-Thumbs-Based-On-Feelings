import torch
import os
from ultralytics import YOLO
import cv2
torch.cuda.empty_cache()
def get_device():
    if torch.cuda.is_available():
        return "0"  
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  
    else:
        return "cpu"
  
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
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                
                label = f"Face: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            save_path = os.path.join(save_directory, image)
            cv2.imwrite(save_path, img)