import os
import cv2
import numpy as np
from PIL import Image
import shutil


recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'
processed_dir_path = 'picture_database'

# Function to get images with their IDs
def get_image_with_id(path):
    os.makedirs(processed_dir_path, exist_ok=True)
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    ids = []
    
    for single_image_path in image_paths:
        face_img = Image.open(single_image_path).convert('L')
        face_np = np.array(face_img, np.uint8)    
        id = int(os.path.split(single_image_path)[-1].split('.')[1])
        faces.append(face_np)
        ids.append(id)
        cv2.waitKey(100)

        student_dir_path = os.path.join(processed_dir_path, str(id))
        os.makedirs(student_dir_path, exist_ok=True)

        shutil.copy(single_image_path, student_dir_path)

    return np.array(ids), faces