'''
KNNClassifier.py

Class for performing k-nearest neighbor classification.

'''

import numpy as np


class KNNClassifier:

    def __init__(self, neighbors=5):
        '''
        Constructor for KNNClassifier.

        Args (optional):
            neighbors (int): The number of neighbors to compare against.
                (Default 5, must be odd)
        '''

        # TODO: Make these sort of statements use conditional
        if neighbors % 2 == 1:
            self.neighbors = neighbors
        else:
            raise RuntimeError('Number of neighbors should be odd.')

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

        nearest = list()

        for instance in self.samples:

            distance = self.distance(instance[1], new_features)

            if len(nearest) < self.neighbors or distance < nearest[-1][1]:
                # If a nearest neighbor, insertion sort into list of neighbors

                idx = 0
                for i, neighbor in enumerate(nearest):
                    if neighbor[1] > distance:
                        idx = i
                        break

                nearest.insert( idx, (instance[0], distance))

                if len(nearest) > self.neighbors:
                    nearest.pop(-1)

        # Determine the majority class

        count = [0] * self.num_classes

        for neighbor in nearest:
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

    classifier = KNNClassifier(neighbors=5)

    for i in range(0, 10):
        classifier.add_sample(0, gen_class0())
        classifier.add_sample(1, gen_class1())
        classifier.add_sample(2, gen_class2())

    print 'Class 0 classified as {}'.format(classifier.classify(gen_class0()))
    print 'Class 1 classified as {}'.format(classifier.classify(gen_class1()))
    print 'Class 2 classified as {}'.format(classifier.classify(gen_class2()))

