import os
import shutil
from sklearn.model_selection import train_test_split

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