
import cv2
import numpy as np

from FaceDetectionModel import FaceDetectionModel

class FaceDetector:
    
    def __init__(self, face_img_data, non_face_img_data, iterations, img_size):
        '''
        Constructor for FaceDetector.

        Args:
            face_img_data (list): Face images.
            non_face_img_data (list): Non-face images.
            iterations (int): Number of AdaBoost iterations.
            img_size (tuple<int, int>): Image dimension.
        '''
        
        self.face_img_data = face_img_data
        self.non_face_img_data = non_face_img_data
        self.model = FaceDetectionModel(iterations, img_size)
    
    def train(self):
        self.model.train(self.face_img_data, self.non_face_img_data)
    
    def classify(self, img_data):
        return self.model.classify(img_data)
