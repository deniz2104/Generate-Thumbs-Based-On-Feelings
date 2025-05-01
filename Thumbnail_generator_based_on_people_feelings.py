from process_frames import *
from prepare_yolo import *
from detect_faces_preprocessing_with_mediapipe import detect_faces
from eliminating_faces_with_motion import *
from detect_faces_with_eyes_closed import *
from get_predominant_person import *
##TODO: Eliminate photos that does not have the predominant character
##logica mea e urmatoarea: elimin cele care sunt neclare din start,masor cu fft,apoi daca nu cu laplacian modificat

if __name__ == '__main__':
    #video_to_frames_one_per_second(video_path='The Present.mp4', frames_dir='test_frames', overwrite=True)
    #detect_faces(image_directory='test_frames',save_directory='test_frames_faces')
    #weight_path = train_yolo_model(data_yaml='widerface_yolo.yaml', img_size=320, batch_size=16, epochs=5)
    #detect_faces_yolo(weight_path="widerface_yolo/yolo_model/weights/best.pt", image_directory='test_frames_faces', save_directory='test_frames_faces_yolo',conf_thresh=0.6)
    #eliminate_photos_with_motion(image_directory='test_frames_faces_yolo', fft_threshold=140, radius=60,save_directory='final_frames')
    #eliminate_photos_with_motion(image_directory='final_frames', fft_threshold=140, radius=60)
    #eliminate_closed_eyes(
    #    image_directory='final_frames',
    #    ear_threshold=0.21  # Lower for animated characters
    #    save_directory='final_frames_closed_eyes
    #)
    pass