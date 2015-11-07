'''
KNNClassifier.py

Class for performing k-nearest neighbor classification.

'''

import numpy as np


class KNNClassifier:

    def __init__(self, k_neighbors=5):
        '''
        Constructor for KNNClassifier.

        Args (optional):
            k_neighbors (int): The number of neighbors to compare against.
                (Default 5, must be odd)
        '''

        # TODO: Make these sort of statements use conditional
        if k_neighbors % 2 == 1:
            self.k_neighbors = k_neighbors
        else:
            self.k_neighbors = 5

        self.reset()


    def reset(self):
        '''
        Resets training on the classifier.
        '''

        self.samples = list()
        self.num_classes = 0


    def add_sample(self, label, features):
        '''
        Adds a sample to the training set.

        Args:
            label (int): The class that this instance belongs to.
            data (numpy.ndarray): The feature vector for this instance.
        '''

        self.samples.append( (label, features) )
        self.num_classes = max(self.num_classes, label+1)


    def distance(self, vec1, vec2):
        '''
        Computes the l2 distance between two vectors.

        Args:
            vec1, vec2 (numpy.ndarray): Vectors to find the distance between.
        Returns:
            float, the l2 distance between two vectors.
        '''

        return np.sqrt( np.sum( (vec1-vec2)**2 ) )


    def classify(self, new_features):
        '''
        Classifies a given feature vector.

        Args:
            new_features (numpy.ndarray): The feature vector to classify.
        Returns:
            int, the class that this feature vector best belongs to.
        '''

        # Calculate distances and Find nearest neighbors

        neighbors = list()

        for instance in self.samples:

            distance = self.distance(instance[1], new_features)

            if len(neighbors) < self.k_neighbors or distance < neighbors[-1][1]:
                # If a nearest neighbor, insertion sort into list of neighbors

                idx = 0
                for n in neighbors:
                    if n[1] > distance:
                        break
                    idx += 1

                neighbors.insert( idx, (instance[0], distance))

                if len(neighbors) > self.k_neighbors:
                    neighbors.pop(-1)

        # Determine the majority class

        count = [0] * self.num_classes

        for neighbor in neighbors:
            count[ neighbor[0] ] += 1

        best_class = 0
        max_count = 0

        for i in range(0, self.num_classes):
            if count[i] > max_count:
                max_count = count[i]
                best_class = i

        return best_class


# Command-Line Invocation

if __name__ == '__main__':
    ''' Tests the kNN classifier with toy data. '''

    # Definitions for toy data classes
    def gen_class0():
        return np.array([1,2,3,4,5]) + np.random.rand(5)
    def gen_class1():
        return np.array([5,4,3,2,1]) + np.random.rand(5)
    def gen_class2():
        return np.array([2,2,2,2,2]) + np.random.rand(5)

    # Train kNN classifier with toy data

    classifier = KNNClassifier(k_neighbors=5)

    for i in range(0, 10):
        classifier.add_sample(0, gen_class0())
        classifier.add_sample(1, gen_class1())
        classifier.add_sample(2, gen_class2())

    print 'Class 0 classified as {}'.format(classifier.classify(gen_class0()))
    print 'Class 1 classified as {}'.format(classifier.classify(gen_class1()))
    print 'Class 2 classified as {}'.format(classifier.classify(gen_class2()))

