'''
CLI for testing the face detector and recognizer

'''

from facerec import ImageIO
from facerec.FaceRecognizer import FaceRecognizer

import argparse, os, random, sys, time


# "Enum" to specify different face recognition datasets
class FaceRecData:
    Yalefaces_A = 'Yalefaces A'
    Yalefaces_B = 'Yalefaces B+'


class FaceRecTest:

    def __init__(self, dataset=FaceRecData.Yalefaces_A, data_directory='data',
            trn_part=70, dev_part=10, tst_part=20, loud=False):
        '''
        Constructor for FaceRecTest.

        Args (optional):
            dataset (FaceRecData.value): The dataset to use (default
                FaceRecData.Yalefaces_A)
            data_directory (str): The directory where the data lives (default
                'data')
            trn_part (int): What percentage of the data should be used for
                training (default 70)
            dev_part (int): "" for tuning (default 10)
            tst_part (int): "" for testing (default 20)
            loud (bool): Whether the classifier should print out fine details
        '''

        self.dataset         = dataset
        self.data_directory  = data_directory
        self.trn_part        = trn_part
        self.dev_part        = dev_part
        self.tst_part        = tst_part
        self.loud            = loud

        self.face_recognizer = FaceRecognizer()

        self.instances       = None
        self.trn_data        = None
        self.dev_data        = None
        self.tst_data        = None

        if sum((trn_part, dev_part, tst_part)) != 100:
            raise RuntimeError('Train/Dev/Test partitions don\'t add up '
                    'to 100%')


    def load_data(self):
        '''
        Loads data to use in training.
        '''

        # Load data

        if self.dataset == FaceRecData.Yalefaces_A:
            self.instances = ImageIO.loadYalefacesImages(self.data_directory)
        elif self.dataset == FaceRecData.Yalefaces_B:
            self.instances = ImageIO.loadExtendedCroppedYalefaces(
                    self.data_directory)
        else:
            raise RuntimeError('FaceRecTest not assigned a valid dataset')

        # Sample from each class to create train/dev/test partitions

        self.trn_data = list()
        self.dev_data = list()
        self.tst_data = list()

        # Sort instances by label
        classes = dict()
        for instance in self.instances:
            label = instance[0]
            if label not in classes.keys():
                classes[label] = list()
            classes[label].append(instance)

        num_labels = max(classes.keys())+1

        # Calculate how many samples should be in each partition
        num_trn = int( (self.trn_part/100.0) * len(self.instances) )
        num_dev = int( (self.dev_part/100.0) * len(self.instances) )
        num_tst = int( (self.tst_part/100.0) * len(self.instances) )

        # If any leftovers, give more to training partition
        num_trn += len(self.instances) - sum((num_trn, num_dev, num_tst))

        # Partition trn/dev/tst

        def partition(src, dest, num):

            num_labels = max(src.keys())+1
            count = 0

            i = 0
            while i < num:
                instances = src[count % num_labels]
                if len(instances) > 0:
                    idx = random.randint(0, len(instances)-1)
                    dest.append( instances.pop(idx) )
                    i += 1
                count += 1

        partition(classes, self.trn_data, num_trn)
        partition(classes, self.dev_data, num_dev)
        partition(classes, self.tst_data, num_tst)


    def train(self):
        '''
        Trains the face recognizer with the training data.
        '''

        if self.trn_data is None:
            raise RuntimeError('Data has not been loaded yet!')

        self.face_recognizer.train(self.trn_data)


    def tune(self):
        '''
        Tunes the face recognizer with the dev data.
        '''

        results = list()

        optimal_d = 1
        optimal_k = 1

        accuracy = 0

        # TODO: Put tuning on FaceRecognizer?

        for d in range(10, 100, 10):

            self.face_recognizer.set_dimensions(d)

            for k in range(1, 21, 2):

                self.face_recognizer.set_k_neighbors(k)

                test_results = self.test(use_dev=True)

                results.append( (d, k, test_results['accuracy']))

                if test_results['accuracy'] > accuracy:
                    optimal_d = d
                    optimal_k = k
                    accuracy = test_results['accuracy']

        self.face_recognizer.set_dimensions(optimal_d)
        self.face_recognizer.set_k_neighbors(optimal_k)

        return results


    def test(self, use_dev=False):
        '''
        Tests the data and prints the results.

        Returns:
            dict, database of results:
                'accuracy'    : Accuracy of the classifier.
                'correct'     : The number the classifier got correct.
                'predictions' : List of predictions the classifier made.
        '''

        if self.tst_data is None:
            raise RuntimeError('Data has not been loaded yet!')

        predicted_labels = list()

        data = self.tst_data if not use_dev else self.dev_data

        correct_count = 0
        for idx, instance in enumerate(data):
            predicted_label = self.face_recognizer.classify(instance[1])
            if predicted_label == instance[0]:
                correct_count += 1
            predicted_labels.append(predicted_label)

        return { 'accuracy'    : correct_count/float(len(data)),
                 'correct'     : correct_count,
                 'predictions' : predicted_labels
        }


    def run(self):
        '''
        Loads, trains, and tests the FaceRecognizer. Prints out a report.
        '''

        print '| ---- ---- FaceRec ---- ----'
        print '| Dataset        : {}'.format(self.dataset)
        print '| Train/Dev/Test : {}/{}/{}'.format(self.trn_part,
                self.dev_part, self.tst_part)

        self.load_data()

        print '|'
        print '| Total Samples  : {}'.format(len(self.instances))
        print '| Dimensions     : {}'.format(len(self.instances[0][1]))
        print '|'

        self.train()
        tune_results = self.tune()

        print '| PCA Dimensions : {}'.format(
                self.face_recognizer.pca_model.k_rank)
        print '| k-Neighbors    : {}'.format(
                self.face_recognizer.knn_classifier.k_neighbors)
        print ''

        test_results = self.test()

        print 'Accuracy: {:.3f} ({}/{})'.format(test_results['accuracy'],
                test_results['correct'], len(self.tst_data))

        if self.loud:
            print '\nTuning results:'
            for entry in tune_results:
                print '\td = {:2d}, k = {:2d}, accuracy = {:.3f}'.format(
                        entry[0], entry[1], entry[2])
            print '\nPer-case results:'
            for idx, predicted_label in enumerate(test_results['predictions']):
                print '\t[Case {:03d}] Predicted {:2d} as {:2d}'.format(
                        idx, self.tst_data[idx][0], predicted_label)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='An end-to-end face '
            'detection and recognition system.')

    parser.add_argument(
        '-recdata',
        type=str,
        required=True,
        help='Where the face recognition data is located.'
    )
    parser.add_argument(
        '-extended',
        action='store_true',
        help='Specifies that the extended Yalefaces dataset should be used '
             'for the face recognizer.',
    )
    parser.add_argument(
        '-loud',
        action='store_true',
        help='When enabled, additional information will be printed.'
    )

    args = parser.parse_args()

    dataset = FaceRecData.Yalefaces_A if not args.extended \
            else FaceRecData.Yalefaces_B

    fr_test = FaceRecTest(
        dataset        = dataset,
        data_directory = args.recdata,
        loud           = args.loud,
    )
    fr_test.run()


