
class HaarFeature:
    
    def __init__(self, feature_type, position, width, height):
        self.type = feature_type
        self.top_left_coord = position
        self.bottom_right_coord = (position[0] + height - 1, position[1] + width - 1)
        self.width = width
        self.height = height
    
    def getRectangleSum(self, integral_image, top_left_coord, bottom_right_coord):
        bottom_right_value = integral_image[bottom_right_coord[0]][bottom_right_coord[1]]
        bottom_left_value = integral_image[bottom_right_coord[0]][top_left_coord[1] - 1]
        
        top_right_value = integral_image[top_left_coord[0] - 1][bottom_right_coord[1]]
        top_left_value = integral_image[top_left_coord[0] - 1][top_left_coord[1] - 1]
        
        return bottom_right_value - top_right_value + top_left_value - top_left_value;
    
    def getFeature1(self, integral_image):
        '''
        Haar feature consisting of two vertical rectangles
        '''
        
        white_top_left_coord = self.top_left_coord
        white_bottom_right_coord = (self.bottom_right_coord[0], self.top_left_coord[1] + (self.width / 2) - 1);
        
        black_top_left_coord = (self.top_left_coord[1], self.top_left_coord[1] + (self.width / 2))
        black_bottom_right_coord = self.bottom_right_coord
        
        white_pixels_sum = self.getRectangleSum(integral_image, white_top_left_coord, white_bottom_right_coord)
        black_pixels_sum = self.getRectangleSum(integral_image, black_top_left_coord, black_bottom_right_coord)
        
        return black_pixels_sum - white_pixels_sum
    
    def getFeature2(self, integral_image):
        '''
        Haar feature consisting of two horizontal rectangles
        '''
        
        black_top_left_coord = self.top_left_coord
        black_bottom_right_coord = (self.top_left_coord[0] + (self.height / 2) - 1, self.bottom_right_coord[1]);
        
        white_top_left_coord = (self.top_left_coord[0] + (self.height / 2), self.top_left_coord[1])
        white_bottom_right_coord = self.bottom_right_coord
        
        white_pixels_sum = self.getRectangleSum(integral_image, white_top_left_coord, white_bottom_right_coord)
        black_pixels_sum = self.getRectangleSum(integral_image, black_top_left_coord, black_bottom_right_coord)
        
        return black_pixels_sum - white_pixels_sum
    
    def getFeature3(self, integral_image):
        '''
        Haar feature consisting of three vertical rectangles
        '''
        
        black_top_left_coord = (self.top_left_coord[0], self.top_left_coord[1] + (self.width / 3))
        black_bottom_right_coord = (self.bottom_right_coord[0], self.bottom_right_coord[1] - (self.width / 3));
        
        white_top_left_coord1 = self.top_left_coord
        white_bottom_right_coord1 = (self.bottom_right_coord[0], self.bottom_right_coord[1] - (2 * self.width / 3));
        
        white_top_left_coord2 = (self.top_left_coord[0], self.top_left_coord[1] + (2 * self.width / 3))
        white_bottom_right_coord2 = self.bottom_right_coord
        
        white_pixels_sum1 = self.getRectangleSum(integral_image, white_top_left_coord1, white_bottom_right_coord1)
        white_pixels_sum2 = self.getRectangleSum(integral_image, white_top_left_coord2, white_bottom_right_coord2)
        black_pixels_sum = self.getRectangleSum(integral_image, black_top_left_coord, black_bottom_right_coord)
        
        return black_pixels_sum - white_pixels_sum1 - white_pixels_sum2
    
    def getFeature4(self, integral_image):
        '''
        Haar feature consisting of single checker board
        '''
        
        white_top_left_coord1 = self.top_left_coord
        white_bottom_right_coord1 = (self.bottom_right_coord[0] - (self.height / 2), self.bottom_right_coord[1] - (self.width / 2));
        
        white_top_left_coord2 = (self.top_left_coord[0] + (self.height / 2), self.top_left_coord[1] + (self.width / 2))
        white_bottom_right_coord2 = self.bottom_right_coord
        
        black_top_left_coord1 = (self.top_left_coord[0], self.top_left_coord[1] + (self.width / 2))
        black_bottom_right_coord1 = (self.bottom_right_coord[0] - (self.height / 2), self.bottom_right_coord[1]);
        
        black_top_left_coord2 = (self.top_left_coord[0] + (self.height / 2), self.top_left_coord[1])
        black_bottom_right_coord2 = (self.bottom_right_coord[0], self.bottom_right_coord[1] - (self.width / 2));
        
        white_pixels_sum1 = self.getRectangleSum(integral_image, white_top_left_coord1, white_bottom_right_coord1)
        white_pixels_sum2 = self.getRectangleSum(integral_image, white_top_left_coord2, white_bottom_right_coord2)
        black_pixels_sum1 = self.getRectangleSum(integral_image, black_top_left_coord1, black_bottom_right_coord1)
        black_pixels_sum2 = self.getRectangleSum(integral_image, black_top_left_coord2, black_bottom_right_coord2)
        
        return black_pixels_sum1 + black_pixels_sum2 - white_pixels_sum1 - white_pixels_sum2
    
    def getFeatureValue(self, integral_image):
        if self.type == 1:
            return self.getFeature1(integral_image)
        elif self.type == 2:
            return self.getFeature2(integral_image)
        elif self.type == 3:
            return self.getFeature3(integral_image)
        else:
            return self.getFeature4(integral_image)
        
    def getClassification(self, integral_image, polarity, threshold):
        featureValue = self.getFeatureValue(integral_image);
        
        if polarity * featureValue < polarity * threshold:
            return 1 # positive example
        else:
            return 0 # negative example