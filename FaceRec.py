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
            part=[70, 10, 20], loud=False, d_tuning=[25, 100, 25],
            k_tuning=[1, 15], var_tuning=[1000, 10000, 1000],
            skip_tuning=False, d_value=0, k_value=3, var_value=10000,
            use_kernel=False): # TODO: Better way to init this?
        '''
        Constructor for FaceRecTest.

        Args (optional):
            dataset (FaceRecData.value): The dataset to use.
            data_directory (str): The directory where the data lives.
            part (int): What percentage of the data should be used for
                training, dev, and test.
            loud (bool): Whether the classifier should print out fine details.
            d_tuning (tuple<int>): The range of PCA dimensions to observe
                during tuning.
            k_tuning (tuple<int>): The ranke of k for the kNN classifier to
                observe during tuning.
        '''

        self.dataset         = dataset
        self.data_directory  = data_directory
        self.trn_part        = part[0]
        self.dev_part        = part[1]
        self.tst_part        = part[2]
        self.loud            = loud
        self.d_tuning        = d_tuning
        self.k_tuning        = k_tuning
        self.var_tuning      = var_tuning
        self.skip_tuning     = skip_tuning
        self.d_value         = d_value
        self.k_value         = k_value
        self.var_value       = var_value
        self.use_kernel      = use_kernel

        self.face_recognizer = FaceRecognizer(use_kernel=use_kernel)

        self.instances       = None
        self.trn_data        = None
        self.dev_data        = None
        self.tst_data        = None

        if sum(part) != 100:
            raise RuntimeError('Train/Dev/Test partitions don\'t add up '
                    'to 100%')
        if len(d_tuning) != 3:
            raise RuntimeError('3 parameters are required for tuning d '
                    '({} found)'.format(len(d_tuning)))
        if len(k_tuning) != 2:
            raise RuntimeError('2 parameters are required for tuning k '
                    '({} found)'.format(len(d_tuning)))


    def load_data(self):
        '''
        Loads data to use in training.
        '''

        # Load data

        if self.dataset == FaceRecData.Yalefaces_A:
            self.instances = ImageIO.loadYalefacesImages(
                    self.data_directory, loud=False)
        elif self.dataset == FaceRecData.Yalefaces_B:
            self.instances = ImageIO.loadExtendedCroppedYalefaces(
                    self.data_directory, loud=False)
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

        self.face_recognizer.set_dimensions(self.d_value)
        self.face_recognizer.set_k_neighbors(self.k_value)
        self.face_recognizer.set_kernel_variance(self.var_value)

        self.face_recognizer.train(self.trn_data)


    def tune(self):
        '''
        Tunes the face recognizer with the dev data.
        '''

        results = list()

        optimal_d = 1
        optimal_k = 1
        optimal_v = 1

        accuracy = 0

        d_start, d_end, d_step = self.d_tuning
        k_start, k_end = self.k_tuning
        v_start, v_end, v_step = self.var_tuning
        if not self.use_kernel:
            v_start, v_end, v_step = 0, 0, 1

        for v in range(v_start, v_end+1, v_step):

            self.face_recognizer.set_kernel_variance(v)

            for d in range(d_start, d_end+1, d_step):

                self.face_recognizer.set_dimensions(d)

                for k in range(k_start, k_end+1, 2):

                    self.face_recognizer.set_k_neighbors(k)

                    test_results = self.test(use_dev=True)

                    results.append( (d, k, test_results['accuracy']))

                    if test_results['accuracy'] > accuracy:
                        optimal_d = d
                        optimal_k = k
                        optimal_v = v
                        accuracy = test_results['accuracy']

        self.face_recognizer.set_dimensions(optimal_d)
        self.face_recognizer.set_k_neighbors(optimal_k)
        self.face_recognizer.set_kernel_variance(optimal_v)

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

        return {
            'accuracy'    : correct_count/float(len(data)),
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

        self.train()

        if not self.skip_tuning:
            print '|'
            print '| Tuning d from {:3d} to {:3d} with step {}'.format(
                    self.d_tuning[0], self.d_tuning[1], self.d_tuning[2])
            print '| Tuning k from {:3d} to {:3d} with step 2'.format(
                    self.k_tuning[0], self.k_tuning[1])
            if self.use_kernel:
                print '| Tuning RBF variance from {} to {} with step {}' \
                        .format(self.var_tuning[0], self.var_tuning[1],
                        self.var_tuning[2])

            tune_results = self.tune()

        print '|'
        print '| PCA Dimensions : {}'.format(
                self.face_recognizer.pca_model.dimensions)
        print '| k-Neighbors    : {}'.format(
                self.face_recognizer.knn_classifier.neighbors)
        if self.use_kernel:
            print '| RBF Variance   : {}'.format(
                    self.face_recognizer.pca_model.variance)
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
        help='Where the face recognition data is located.',
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
        help='When enabled, additional information will be printed.',
    )
    parser.add_argument(
        '-tuning_partition', '-part',
        nargs=3,
        type=int,
        default=[70, 10, 20],
        help='How much the datset should be partitioned between training, '
             'dev, and testing.',
    )
    parser.add_argument(
        '-d_tuning',
        nargs=3,
        type=int,
        default=[0, 0, 1],
        help='The number of PCA dimensions to observe during tuning '
             '(start/end/step).',
    )
    parser.add_argument(
        '-k_tuning',
        nargs=2,
        type=int,
        default=[1, 5],
        help='The value of k for the kNN classifier to observe during '
             'tuning (start/end)',
    )
    parser.add_argument(
        '-var_tuning',
        nargs=3,
        type=int,
        default=[2500, 10000, 2500],
        help='The values for the RBF variance to observe during tuning '
             '(start/end/step)'
    )
    parser.add_argument(
        '-skip_tuning',
        action='store_true',
        help='When enabled, tuning parameters will be skipped.',
    )
    parser.add_argument(
        '-k_value',
        type=int,
        default=3,
        help='Use this value for k when skipping tuning.',
    )
    parser.add_argument(
        '-d_value',
        type=int,
        default=0,
        help='Use this value for d when skipping tuning.',
    )
    parser.add_argument(
        '-var_value',
        type=float,
        default=10000,
        help='Use this value for the RBF variance when skipping tuning.',
    )
    parser.add_argument(
        '-use_kernel',
        action='store_true',
        help='Use kernel PCA instead of linear PCA.',
    )

    args = parser.parse_args()

    dataset = FaceRecData.Yalefaces_A if not args.extended \
            else FaceRecData.Yalefaces_B

    fr_test = FaceRecTest(
        dataset        = dataset,
        data_directory = args.recdata,
        part           = args.tuning_partition,
        loud           = args.loud,
        d_tuning       = args.d_tuning,
        k_tuning       = args.k_tuning,
        var_tuning     = args.var_tuning,
        skip_tuning    = args.skip_tuning,
        d_value        = args.d_value,
        k_value        = args.k_value,
        var_value      = args.var_value,
        use_kernel     = args.use_kernel,
    )
    fr_test.run()


