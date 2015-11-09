
import cv2
import numpy as np

from ViolaJonesModel import ViolaJonesModel

class FaceDetector:
    
    def __init__(self, iterations, img_size):
        '''
        Constructor for FaceDetector.

        Args:
            face_img_data (list): Face images.
            img_size (tuple<int, int>): Image dimension.
        '''
        
        self.model = ViolaJonesModel(iterations, img_size)
    
    def train(self, face_img_data, non_face_img_data):
        self.model.fit(face_img_data, non_face_img_data)
    
    def classify(self, img_data):
        return self.model.classify(img_data)
