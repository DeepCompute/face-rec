
import cv2
import numpy as np
import os

from FaceDetectionModel import FaceDetectionModel
from Image import Image

def loadImages(path, label):
    images = []
    for file in os.listdir(path):
        if file.endswith('.pgm'):
            image = cv2.imread(os.path.join(path, file), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            integral_image = cv2.integral(image)
            images.append(Image(integral_image, label))
    return images

if __name__ == '__main__':
    path = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\face.train.tar\\face.train\\train\\'
    
    face_images = [];
    non_face_images = [];

    face_images.append(loadImages(path + 'face', 1))
    non_face_images.append(loadImages(path + 'non-face', -1))
    
    model = FaceDetectionModel()
    model.fitModel(face_images, non_face_images)
    
    print ""