
import cv2
import numpy as np

from FaceDetectionModel import FaceDetectionModel

class FaceDetector:
    
    def __init__(self, face_img_data, non_face_img_data):
        self.face_img_data = face_img_data
        self.non_face_img_data = non_face_img_data
        self.model = FaceDetectionModel(10)
    
    def fitModel(self):
        self.model.fitModel(self.face_img_data, self.non_face_img_data)
    
    def classify(self, img_data):
        self.model.classify(img_data)
