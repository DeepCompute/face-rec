'''
ImageIO.py

Module with helper functions for loading images.

'''

import cv2
import os


def loadFaceDetectionImages(directory, label):
    '''
    Loads base face detection images from a given directory.

    Args:
        directory (str): The directory where face detection images are.
        label (int): The label that should be assigned to each image.
    Returns:
        list[Image], the loaded images.
    '''
    image_data = []
    for file in os.listdir(directory):
        if file.endswith('.pgm'):
            image = cv2.imread(os.path.join(directory, file), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            integral_image = cv2.integral(image)
            
            img_data_tuple = list()
            img_data_tuple.append(integral_image) # integral image
            img_data_tuple.append(label) # label
            img_data_tuple.append(0.0) # weight
            
            image_data.append(img_data_tuple)
    return image_data


def loadFaceRecognitionImages(directory):
    '''
    Loads base face recognition images from a given directory.

    Args:
        directory (str): The directory where the face recognition images are.
    Returns:
        list[Image], the loaded images.
    '''
    pass

