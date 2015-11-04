'''

Wrapper class for image data

'''

class Image:
    
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.weight = 0 # Used for face detection model
    
    def setWeight(self, weight):
        self.weight = weight
        
        