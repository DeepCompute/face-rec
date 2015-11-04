
class HaarFeature:
    
    def __init__(self, feature_type, position, width, height):
        self.type = feature_type
        self.top_left_coord = position
        self.bottom_right_coord = (position[0] + height - 1, position[1] + width - 1)
        self.width = width
        self.height = height
        self.threshold = 0
        self.polarity = 0
    
    def setThreshold(self, threshold):
        self.threshold = threshold
    
    def setPolarity(self, polarity):
        self.polarity = polarity
    
    def getRectangleSum(integral_image, top_left_coord, bottom_right_coord):
        bottom_right_value = integral_image[bottom_right_coord[0]][bottom_right_coord[1]]
        bottom_left_value = integral_image[bottom_right_coord[0]][top_left_coord[1] - 1]
        
        top_right_value = integral_image[top_left_coord[0] - 1][bottom_right_coord[1]]
        top_left_value = integral_image[top_left_coord[0] - 1][top_left_coord[1] - 1]
        
        return bottom_right_value - top_right_value + top_left_value - top_left_value;
    
    def getFeature1(self, integral_image):
        '''
        Haar feature value consisting of two vertical rectangles
        '''
        
        white_top_left_coord = self.top_left_coord
        white_bottom_right_coord = (self.top_left_coord[0] + self.height - 1, self.top_left_coord[1] + (self.width / 2) - 1);
        
        black_top_left_coord = (self.top_left_coord[1], self.top_left_coord[1] + (self.width / 2))
        black_bottom_right_coord = self.bottom_right_coord
        
        white_pixels_sum = self.getRectangleSum(integral_image, white_top_left_coord, white_bottom_right_coord)
        black_pixels_sum = self.getRectangleSum(integral_image, black_top_left_coord, black_bottom_right_coord)
        
        return white_pixels_sum - black_pixels_sum
    
    def getFeature2(self, integral_image):
        '''
        Haar feature value consisting of two horizontal rectangles
        '''
        
        white_top_left_coord = self.top_left_coord
        white_bottom_right_coord = (self.top_left_coord[0] + self.width - 1, self.bottom_right_coord[1] - (self.height / 2));
        
        black_top_left_coord = (self.top_left_coord[0] + (self.height / 2), self.top_left_coord[1])
        black_bottom_right_coord = self.bottom_right_coord
        
        white_pixels_sum = self.getRectangleSum(integral_image, white_top_left_coord, white_bottom_right_coord)
        black_pixels_sum = self.getRectangleSum(integral_image, black_top_left_coord, black_bottom_right_coord)
        
        return white_pixels_sum - black_pixels_sum
    
    def getFeatureValue(self, integral_image):
        if self.type == 1:
            return self.getFeature1(integral_image)
        else:
            return self.getFeature2(integral_image)
        
    def getClassification(self, integral_image):
        featureValue = self.getFeatureValue(integral_image);
        
        if self.polarity * featureValue < self.polarity * self.threshold:
            return 1 # positive example
        else:
            return 0 # negative example