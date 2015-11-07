'''
FaceRecognizer.py

Class for training face recognition models and performing classification.

'''

import numpy as np

from FaceRecognitionModel import *
from KNNClassifier import *


class FaceRecognizer:

    def __init__(self, k_rank=0, k_neighbors=5):
        '''
        Constructor for FaceRecognizer.

        Args (optional):
            k_rank (int): How many principal components to keep.
            k_neighbors (int): How many neighbors to compare against in the
                kNN classifier.
        '''

        self.face_recognition_model = FaceRecognitionModel(k_rank=0)
        self.knn_classifier = KNNClassifier(k_neighbors=k_neighbors)


    def train(self, instances):
        '''
        Trains the recognizer with a set of faces.

        Args:
            instances (list<tuple<int, numpy.ndarray>>): List of label/face
                data pairs.
        '''

        # Stack all of the faces together and learn principal components

        faces = None

        for instance in instances:
            if faces is None: #TODO: Use proper way for this
                faces = instance[1]
            else:
                faces = np.vstack((faces, instance[1]))
        faces = faces.T

        self.face_recognition_model.fit(faces)

        # Add each class to the kNN classifier

        for instance in instances:
            label, face = instance
            t_face = self.face_recognition_model.transform(face)
            self.knn_classifier.add_sample(label, t_face)


    def classify(self, face):
        '''
        Classifies a given face from the trained set.

        Args:
            face (numpy.ndarray): The face to classify.
        Returns:
            int, the class the face best belongs to.
        '''

        t_face = self.face_recognition_model.transform(face)
        return self.knn_classifier.classify(t_face)


# Command-Line Invocation

if __name__ == '__main__':
    ''' Test FaceRecognizer class with toy data. '''

    # Parameters to test with
    num_classes = 10
    num_samples = 50
    num_features = 1000
    num_tests = 10

    # Function to generate a random feature vector for each class (with some
    # bleed between dimensions)
    def feature_for_label(label):
        return label+num_classes*np.random.rand(num_features)

    # Generate train data

    instances = list()

    for label in range(0, num_classes):
        for i in range(0, num_samples):
            instances.append( (label, feature_for_label(label)) )

    # Train the face recognizer

    print 'Training model...'

    face_recognizer = FaceRecognizer()
    face_recognizer.train(instances)

    # Test accuracy of classifier

    total_correct = float(0)
    total_overall = float(0)

    for label in range(0, num_classes):

        correct = float(0)

        for i in range(0, num_tests):
            features = feature_for_label(label)
            predicted_label = face_recognizer.classify(features)
            if predicted_label == label:
                correct += 1

        total_correct += correct
        total_overall += num_tests

        print 'Accuracy for class {}: {}'.format(label, correct/num_tests)

    print 'Overall accuracy: {}'.format(total_correct/total_overall)


    print 'Done'

