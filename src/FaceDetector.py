
import cv2
import numpy as np
import os

from FaceDetectionModel import FaceDetectionModel
from Image import Image

def loadImageData(path, label):
    image_data = []
    for file in os.listdir(path):
        if file.endswith('.pgm'):
            image = cv2.imread(os.path.join(path, file), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            integral_image = cv2.integral(image)
            
            img_data_tuple = list()
            img_data_tuple.append(integral_image)
            img_data_tuple.append(label)
            img_data_tuple.append(0.0)
            
            image_data.append(img_data_tuple)
    return image_data

if __name__ == '__main__':
    path = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\face.train.tar\\face.train\\train\\'

    face_img_data = loadImageData(path + 'face', 1)
    non_face_img_data = loadImageData(path + 'non-face', 0)
    
    model = FaceDetectionModel()
    model.fitModel(face_img_data, non_face_img_data)
    
    print ""