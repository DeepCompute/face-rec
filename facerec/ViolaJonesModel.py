
import cv2
import numpy as np

from HaarFeature import HaarFeature

class ViolaJonesModel: 
    
    def __init__(self, iterations, img_size):
        '''
        Constructor for ViolaJonesModel.

        Args:
            iterations (int): Number of AdaBoost iterations.
            img_size (tuple<int, int>): Image dimension.
        '''
        
        self.features = None
        self.weights = None
        self.threshold = 0
        self.iterations = iterations
        self.img_size = img_size
        
    def fit(self, face_img_data, non_face_img_data):
        # Initialize weights
        
        num_of_face_images = len(face_img_data)
        num_of_non_face_images = len(non_face_img_data)
        
        face_image_weight = 1.0 / (2 * num_of_face_images)
        non_face_image_weight = 1.0 / (2 * num_of_non_face_images)
        
        total_weight = 0.0;
        
        for face_image in face_img_data:
            face_image[2] = face_image_weight
            total_weight = total_weight + face_image_weight
        
        for non_face_image in non_face_img_data:
            non_face_image[2] = non_face_image_weight
            total_weight = total_weight + non_face_image_weight
        
        image_height = self.img_size[0]
        image_width = self.img_size[1]
        
        # Ideally these values should be 0 but they dramatically increase training time
        min_height = int(float(image_height) * 0.7)
        min_width_1 = int(float(image_height) * 0.7)
        min_width_2 = int(float(image_height) * 0.4)
        
        if min_height % 2 == 1:
            min_height = min_height + 1
        
        if min_width_1 % 2 == 1:
            min_width_1 = min_width_1 + 1
        
        if min_width_2 % 2 == 0:
            min_width_2 = min_width_2 + 1   
        
        haar_features = [];
        
        # Create features
        for row in range(1, image_height, 1):
            for col in range(1, image_width, 1):
                for height in range(min_height, image_height - row, 2):
                    for width in range(min_width_1, image_width - col, 2):
                        haar_features.append(HaarFeature(1, (row, col), width, height))
                        haar_features.append(HaarFeature(2, (row, col), width, height))
                        haar_features.append(HaarFeature(4, (row, col), width, height))
                    for width in range(min_width_2, image_width - col, 3):
                        haar_features.append(HaarFeature(3, (row, col), width, height))
                    
        self.features = dict()
        img_weights = []
            
        for t in range(self.iterations):
            # Re-normalize weights
            
            for face_image in face_img_data:
                face_image[2] = face_image[2] / total_weight
        
            for non_face_image in non_face_img_data:
                non_face_image[2] = non_face_image[2] / total_weight
            
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
                    
                    if left_error < right_error:
                        error = left_error
                        polarity = -1
                    else:
                        error = right_error
                    
                    if error < minimum_error:
                        minimum_error = error
                        self.features[t] = (haar_feature, polarity, feature_value_dict[sorted_index]) # feature, polarity, threshold
                    
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
                predicted_label = best_feature[0].getClassification(face_image[0], best_feature[1], best_feature[2])
                classification_error = classification_error + (face_image[2] * (1 - predicted_label))
                face_image[1] = predicted_label
        
            for non_face_image in non_face_img_data:
                predicted_label = best_feature[0].getClassification(non_face_image[0], best_feature[1], best_feature[2])
                classification_error = classification_error + (non_face_image[2] * predicted_label)
                non_face_image[1] = predicted_label
            
            # Compute beta
            img_weights.append(classification_error / (1.0 - classification_error))
            
            total_weight = 0.0
            
            # Update weights
            for face_image in face_img_data:
                if face_image[1] == 1:
                    face_image[2] = face_image[2] * img_weights[t]
                total_weight = total_weight + face_image[2]
        
            for non_face_image in non_face_img_data:
                if non_face_image[1] == 0:
                    non_face_image[2] = non_face_image[2] * img_weights[t]
                total_weight = total_weight + non_face_image[2]
                
        # Set final weights and threshold used for classification
            
        self.weights = []
        
        sum_of_alphas = 0.0
        
        for t in range(self.iterations):
            alpha = np.log(1.0 / img_weights[t])
            sum_of_alphas = sum_of_alphas + alpha
            self.weights.append(alpha)
            
        self.threshold = 0.5 * sum_of_alphas
    
    def classify(self, image):
        value = 0.0
        
        for t in range(self.iterations):
            best_feature = self.features[t]
            value = value + (self.weights[t] * best_feature[0].getClassification(image[0], best_feature[1], best_feature[2]))
            
        if value >= self.threshold:
            return 1 # positive (i.e. face)
        else:
            return 0
    
    def getThreshold(self):
        return self.threshold
    
    def setThreshold(self, threshold):
        self.threshold = threshold
    
    # This should be defined as abstract method in the future...
    def getDetails(self):
        weights_dict = dict()
        
        for i in range(len(self.weights)):
            weights_dict[i] = self.weights[i]
                
        sorted_indices = sorted(weights_dict, key=weights_dict.get, reverse=True)
        
        details = list()
        details.append('List of learned Haar features (ordered by their weights)...')
        
        for index in sorted_indices:
            feature = self.features[index]
            details.append(feature[0].getDetail())
        
        return details