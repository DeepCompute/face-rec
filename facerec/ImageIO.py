'''
ImageIO.py

Module with helper functions for loading images.

'''

import cv2
import numpy as np
import scipy.ndimage

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


def loadYalefacesImages(directory, loud=True):
    '''
    Loads base face recognition images from a given directory (Yale face
    database).

    Args:
        directory (str): The directory where the face recognition images are.
    Returns:
        list<tuple<int, numpy.ndarray, str>>, the loaded images as
            label/feature/mode tuples.
    '''

    if not directory.endswith('/'):
        directory = directory + '/'

    images = list()

    for i in range(1, 15+1):
        base_filename = directory+'subject{:02d}'.format(i)

        modes = ['centerlight',
                 'glasses',
                 'happy',
                 'leftlight',
                 'noglasses',
                 'normal',
                 'rightlight',
                 'sad',
                 'sleepy',
                 'surprised',
                 'wink',
        ]

        for mode in modes:

            filename = '{}.{}'.format(base_filename, mode)

            try:
                img = scipy.ndimage.imread(filename)
                images.append( (i-1, img.flatten(), mode) )
            except:
                if loud:
                    print 'Warning: [ImageIO] Unable to read {}'.format(
                        filename)

    return images


def loadExtendedCroppedYalefaces(directory, loud=True):
    '''
    Loads images from the cropped extended Yalefaces B+ dataset.

    Args:
        directory (str): The directory where the face recognition images are.
    Returns:
        list<tuple<int, numpy.ndarray>>, the loaded images as label/feature
            tuples.
    '''

    if not directory.endswith('/'):
        directory = directory + '/'

    images = list()

    label = 0

    for folder in os.listdir(directory):

        for filename in os.listdir(directory+folder):
            # Check that it's an image and not the ambient image
            if filename.endswith('.pgm') and 'Ambient' not in filename:

                filepath = directory + folder + '/' + filename

                try:
                    img = scipy.ndimage.imread(filepath)
                    images.append( (label, img.flatten(), filename[-12:-4]) )
                except:
                    if loud:
                        print 'Warning: [ImageIO] Unable to read {}'.format(
                            filepath)

        label += 1

    return images


# Command-line Invocation

if __name__ == '__main__':
    ''' Run IO functions. '''

    print 'Loading Yalefaces (A)'
    loadYalefacesImages('data/yalefaces/')
    print 'Loading Yalefaces (Extended)'
    loadExtendedCroppedYalefaces('data/yalefaces-ext/')
    print 'Done'

