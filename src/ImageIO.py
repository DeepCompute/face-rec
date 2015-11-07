'''
ImageIO.py

Module with helper functions for loading images.

'''

import numpy as np
import scipy.ndimage


def loadFaceDetectionImages(directory):
    '''
    Loads base face detection images from a given directory.

    Args:
        directory (str): The directory where face detection images are.
    Returns:
        list[Image], the loaded images.
    '''
    pass


def loadFaceRecognitionImages(directory):
    '''
    Loads base face recognition images from a given directory (Yale face
    database).

    Args:
        directory (str): The directory where the face recognition images are.
    Returns:
        list<tuple<int, numpy.ndarray>>, the loaded images as label/feature
            tuples.
    '''

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
                images.append( (i-1, img.flatten()) )
            except:
                print 'Warning: [ImageIO] Unable to read {}'.format(filename)

    return images



# Command-line Invocation

if __name__ == '__main__':
    ''' Run IO functions. '''

    print 'Loading...'
    #loadFaceDetectionImages('data/?')
    loadFaceRecognitionImages('data/yalefaces/')
    print 'Done'

