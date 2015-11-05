
import cv2
import numpy as np

from HaarFeature import HaarFeature

class FaceDetectionModel:
    
    def __init__(self, iterations):
        self.features = None
        self.weights = None
        self.iterations = iterations
        
    def fitModel(self, face_img_data, non_face_img_data):
        # Initialize weights
        
        num_of_face_images = len(face_img_data)
        num_of_non_face_images = len(non_face_img_data)
        
        face_image_weight = 1.0 / (2 * num_of_face_images)
        non_face_image_weight = 1.0 / (2 * num_of_non_face_images)
        
        sum_of_weights = 0.0;
        
        for face_image in face_img_data:
            face_image[2] = face_image_weight
        
        for non_face_image in non_face_img_data:
            non_face_image[2] = non_face_image_weight
        
        image_height = 19
        image_width = 19
        
        haar_features = [];
        
        # Create features
        for row in range(1, image_height, 1):
            for col in range(1, image_width, 1):
                for height in range(14, image_height - row, 2):
                    for width in range(14, image_width - col, 2):
                        haar_features.append(HaarFeature(1, (row, col), width, height))
        
        self.features = dict()
        self.weights = []
            
        for t in range(self.iterations):
            # Re-normalize weights
            
            total_weight = 0.0
            
            for face_image in face_img_data:
                total_weight = total_weight + face_image[2]
        
            for non_face_image in non_face_img_data:
                total_weight = total_weight + non_face_image[2]
            
            for face_image in face_img_data:
                face_image[2] = face_image_weight / total_weight
        
            for non_face_image in non_face_img_data:
                non_face_image[2] = non_face_image_weight / total_weight
            
            # Select best weak learner
        
            minimum_error = float("inf")
            
            for haar_feature in haar_features:
                index = 0
                
                feature_value_dict = dict() # image index : feature value
                
                total_positive_weights = 0.0 # T+
                total_negative_weights = 0.0 # T-
                sum_positive_weights = 0.0 # S+
                sum_negative_weights = 0.0 # S-
            
                for face_image in face_img_data:
                    feature_value_dict[index] = haar_feature.getFeatureValue(face_image[0])
                    total_positive_weights = total_positive_weights + face_image[2]
                    index = index + 1
        
                for non_face_image in non_face_img_data:
                    feature_value_dict[index] = haar_feature.getFeatureValue(non_face_image[0])
                    total_negative_weights = total_negative_weights + non_face_image[2]
                    index = index + 1
                
                sorted_indices = sorted(feature_value_dict, key=feature_value_dict.get, reverse=False)
                
                current_error = float("inf")
                
                for sorted_index in sorted_indices:
                    image_data = None
                    
                    if sorted_index > num_of_face_images - 1:
                        image_data = non_face_img_data[sorted_index - num_of_face_images]
                    else:
                        image_data = face_img_data[sorted_index]
                    
                    # Find min(S+ + (T- - S-), S- + (T+ - S+)
                    left_error = sum_positive_weights + (total_negative_weights - sum_negative_weights)
                    right_error = sum_negative_weights + (total_positive_weights - sum_positive_weights)
                    
                    error = 0.0
                    polarity = 1
                    threshold = feature_value_dict[sorted_index]
                    
                    if left_error < right_error:
                        error = left_error
                        polarity = -1
                    else:
                        error = right_error
                    
                    if error < current_error: 
                        current_error = error
                        
                    if error < minimum_error:
                        minimum_error = error
                        self.features[t] = (haar_feature, polarity, threshold) # feature, polarity, threshold
                    
                    # Update sum of weights
                    
                    if sorted_index > num_of_face_images - 1:
                        # negative sample
                        sum_negative_weights = sum_negative_weights + image_data[2]
                    else:
                        # positive sample
                        sum_positive_weights = sum_positive_weights + image_data[2]
                        
            
            best_feature = self.features[t]
            
            classification_error = 0.0
            
            for face_image in face_img_data:
                label_result = best_feature[0].getClassification(face_image[0], best_feature[1], best_feature[2])
                classification_error = classification_error + (face_image[2] * abs(label_result - 1))
                face_image[1] = label_result
        
            for non_face_image in non_face_img_data:
                label_result = best_feature[0].getClassification(non_face_image[0], best_feature[1], best_feature[2])
                classification_error = classification_error + (non_face_image[2] * label_result)
                non_face_image[1] = label_result
            
            # Compute beta
            self.weights.append(classification_error / (1 - classification_error))
            
            # Update weights
            for face_image in face_img_data:
                if face_image[1] == 1:
                    face_image[2] = face_image[2] * self.weights[t]
        
            for non_face_image in non_face_img_data:
                if non_face_image[1] == 0:
                    non_face_image[2] = non_face_image[2] * self.weights[t]
    
    def classify(self, image_data):
        alphas = []
        
        sum_of_alphas = 0.0
        
        for t in range(self.iterations):
            alpha = np.log(1.0 / self.weights[t])
            sum_of_alphas = sum_of_alphas + alpha
            alphas.append(alpha)
            
        threshold = 0.5 * sum_of_alphas
        
        num_of_matches = 0.0
        
        for image in image_data:
            value = 0.0
            for t in range(self.iterations):
                best_feature = self.features[t]
                value = value + (alphas[t] * best_feature[0].getClassification(image[0], best_feature[1], best_feature[2]))
            
            if value >= threshold:
                num_of_matches = num_of_matches + 1 # Assume all face for now
        
        accuracy = num_of_matches / len(image_data)
        
        print 'Accuracy: ' + str(accuracy * 100)