
import cv2

from HaarFeature import HaarFeature

class FaceDetectionModel:
    
    def __init__(self):
        self.features = None
        self.weights = None
        
    def fitModel(self, face_images, non_face_images):
        # Initialize weights
        
        num_of_face_images = len(face_images)
        num_of_non_face_images = len(non_face_images)
        
        face_image_weight = 1.0 / (2 * num_of_face_images)
        non_face_image_weight = 1.0 / (2 * num_of_non_face_images)
        
        sum_of_weights = 0.0;
        
        for face_image in face_images:
            face_image.setWeight(face_image_weight)
        
        for non_face_image in num_of_non_face_images:
            non_face_image.setWeight(non_face_image_weight)
        
        image_height = 19
        image_width = 19
        
        haar_features = [];
        
        # Create features
        for row in range(1, image_height, 1):
            for col in range(1, image_width, 1):
                for height in range(2, image_height - row, 2):
                    for width in range(2, image_width - col, 2):
                        haar_features.append(HaarFeature(1, (row, col), width, height))
            
        for t in range(20):
            # Re-normalize weights
            
            total_weight = 0.0
            
            for face_image in face_images:
                total_weight = total_weight + face_image.weight
        
            for non_face_image in num_of_non_face_images:
                total_weight = total_weight + non_face_image.weight
            
            for face_image in face_images:
                face_image.setWeight(face_image_weight / total_weight)
        
            for non_face_image in num_of_non_face_images:
                non_face_image.setWeight(non_face_image_weight / total_weight)
            
        
        print ''