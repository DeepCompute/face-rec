'''
CLI for testing the face detector
'''

from facerec import ImageIO
from facerec.FaceDetector import FaceDetector

import argparse, os, random, sys, time, numpy

# "Enum" to specify different face detection data sets
class FaceDetData:
    MIT_Face_Data = 'MIT Face Data'

class FaceDetTest:
    
    def __init__(self, dataset=FaceDetData.MIT_Face_Data, data_directory='data/faces',
            part=[70, 10, 20], iterations=5, desired_accuracy=85):
        '''
        Constructor for FaceDetTest.

        Args (optional):
            dataset (FaceDetData.value): The dataset to use.
            data_directory (str): The directory where the data lives.
            part (int): What percentage of the data should be used for
                training, dev, and test.
            iterations (int): Number of iterations for training.
            desired_accuracy (double): Desired accuracy for face detection.
                There will be a trade-off between false positives rate.
        '''

        self.dataset            = dataset
        self.data_directory     = data_directory
        self.trn_part           = part[0]
        self.dev_part           = part[1]
        self.tst_part           = part[2]
        self.iterations         = iterations
        self.desired_accuracy   = desired_accuracy
        
        self.face_instances     = None
        self.non_face_instances = None
        self.face_trn_data      = None
        self.non_face_trn_data  = None
        self.face_dev_data      = None
        self.non_face_dev_data  = None
        self.face_tst_data      = None
        self.non_face_tst_data  = None
        
        self.total_face_instances = 0
        self.total_non_face_instances = 0

        if sum(part) != 100:
            raise RuntimeError('Train/Dev/Test partitions don\'t add up to 100%')
        if iterations < 1:
            raise RuntimeError('There must be at least 1 iteration for training')
        if desired_accuracy < 0 or desired_accuracy > 100:
            raise RuntimeError('Desired accuracy must be a number between 0 and 100')
    
    def getAccuracy(self, img_data):
        num_of_matches = 0.0
        
        for image in img_data:
            predicted_label = self.face_detector.classify(image)
            
            if image[1] == predicted_label:
                num_of_matches = num_of_matches + 1
    
        accuracy = num_of_matches / len(img_data)
    
        return (accuracy, int(num_of_matches))


    def load_data(self):
        '''
        Loads data to use in training.
        '''
        
        self.face_instances     = list()
        self.non_face_instances = list()

        # Load data

        if self.dataset == FaceDetData.MIT_Face_Data:
            self.face_instances.extend(ImageIO.loadFaceDetectionImages(self.data_directory + '/train/face', 1))
            self.face_instances.extend(ImageIO.loadFaceDetectionImages(self.data_directory + '/test/face', 1))
            self.non_face_instances.extend(ImageIO.loadFaceDetectionImages(self.data_directory + '/train/non-face', 0))
            self.non_face_instances.extend(ImageIO.loadFaceDetectionImages(self.data_directory + '/test/non-face', 0))
        else:
            raise RuntimeError('FaceDetTest not assigned a valid dataset')
        
        img_size = numpy.shape(self.face_instances[0][0])
        
        self.face_detector = FaceDetector(self.iterations, img_size)
        
        self.total_face_instances = len(self.face_instances)
        self.total_non_face_instances = len(self.non_face_instances)

        # Sample from each class to create train/dev/test partitions

        self.face_trn_data      = list()
        self.non_face_trn_data  = list()
        self.face_dev_data      = list()
        self.non_face_dev_data  = list()
        self.face_tst_data      = list()
        self.non_face_tst_data  = list()

        # Calculate how many samples should be in each partition
        num_face_trn        = int( (self.trn_part/100.0) * len(self.face_instances) )
        num_non_face_trn    = int( (self.trn_part/100.0) * len(self.non_face_instances) )
        num_face_dev        = int( (self.dev_part/100.0) * len(self.face_instances) )
        num_non_face_dev    = int( (self.dev_part/100.0) * len(self.non_face_instances) )
        num_face_tst        = int( (self.tst_part/100.0) * len(self.face_instances) )
        num_non_face_tst    = int( (self.tst_part/100.0) * len(self.non_face_instances) )

        # If any leftovers, give more to training partition
        num_face_trn += len(self.face_instances) - sum((num_face_trn, num_face_dev, num_face_tst))
        num_non_face_trn += len(self.non_face_instances) - sum((num_non_face_trn, num_non_face_dev, num_non_face_tst))

        # Partition trn/dev/tst

        def partition(src, dest, num):
            i = 0
            while i < num:
                if len(src) > 0:
                    idx = random.randint(0, len(src)-1)
                    dest.append( src.pop(idx) )
                    i += 1

        partition(self.face_instances, self.face_trn_data, num_face_trn)
        partition(self.non_face_instances, self.non_face_trn_data, num_non_face_trn)
        partition(self.face_instances, self.face_dev_data, num_face_dev)
        partition(self.non_face_instances, self.non_face_dev_data, num_non_face_dev)
        partition(self.face_instances, self.face_tst_data, num_face_tst)
        partition(self.non_face_instances, self.non_face_tst_data, num_non_face_tst)


    def train(self):
        '''
        Trains the face detector with the training data.
        '''

        if self.face_trn_data is None or self.non_face_trn_data is None:
            raise RuntimeError('Data has not been loaded yet!')

        self.face_detector.train(self.face_trn_data, self.non_face_trn_data)


    def tune(self):
        '''
        Tunes the face detector with the dev data.
        '''

        model = self.face_detector.getModel()
        initial_threshold = model.getThreshold()
        new_threshold = initial_threshold
    
        accuracy = self.getAccuracy(self.face_dev_data)[0]
        
        desired_accuracy = float(self.desired_accuracy) / 100.0;
    
        runtime = time.time()
    
        while accuracy < desired_accuracy:
            accuracy = self.getAccuracy(self.face_dev_data)[0]
        
            if accuracy < desired_accuracy:
                new_threshold = new_threshold - 0.01;
            
            model.setThreshold(new_threshold)

        return (initial_threshold, new_threshold)


    def test(self, use_dev=False):
        '''
        Tests the data and prints the results.

        Returns:
            dict, database of results:
                'faces'         : Accuracy of detecting faces.
                'non-faces'     : Accuracy of detecting non-faces.
        '''

        if self.face_tst_data is None or self.non_face_tst_data is None:
            raise RuntimeError('Data has not been loaded yet!')

        face_accuracy = self.getAccuracy(self.face_tst_data)
        non_face_accuracy = self.getAccuracy(self.non_face_tst_data)

        return {
            'faces'         : face_accuracy,
            'non-faces'     : non_face_accuracy
        }


    def run(self):
        '''
        Loads, trains, and tests the FaceDetector. Prints out a report.
        '''

        print '| ---- ---- FaceDet ---- ----'
        print '| Dataset                   : {}'.format(self.dataset)
        print '| Train/Dev/Test            : {}/{}/{}'.format(self.trn_part,
                self.dev_part, self.tst_part)

        self.load_data()

        print '|'
        print '| Total Face Samples        : {}'.format(self.total_face_instances)
        print '| Total Non Face Samples    : {}'.format(self.total_non_face_instances)
        print '| Number of Iterations      : {}'.format(self.iterations)

        self.train()

        print '|'
        print '| Tuning detection accuracy to {}%'.format(self.desired_accuracy)

        tune_results = self.tune()
        
        print '| Threshold reduced from {:.2f} to {:.2f}'.format(tune_results[0], tune_results[1])

        test_results = self.test()

        print '|'
        print '| Accuracy (faces)          : {:.3f} ({}/{})'.format(test_results['faces'][0],
                test_results['faces'][1], len(self.face_tst_data))
        print '| Accuracy (non-faces)      : {:.3f} ({}/{})'.format(test_results['non-faces'][0],
                test_results['non-faces'][1], len(self.non_face_tst_data))
        print '|'
        
        model_details = self.face_detector.getModelDetails()
        
        for model_detail in model_details:
            print '| ' + model_detail

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='An end-to-end face '
            'detection and recognition system.')

    parser.add_argument(
        '-detdata',
        type=str,
        required=True,
        help='Where the face detection data is located.',
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
        '-iterations',
        nargs=1,
        type=int,
        default=5,
        help='The number of iterations for training.',
    )
    parser.add_argument(
        '-desired_accuracy',
        nargs=1,
        type=int,
        default=85,
        help='The desired accuracy for detecting faces in face images.',
    )

    args = parser.parse_args()

    dataset = FaceDetData.MIT_Face_Data

    fd_test = FaceDetTest(
        dataset             = dataset,
        data_directory      = args.detdata,
        part                = args.tuning_partition,
        iterations          = args.iterations,
        desired_accuracy    = args.desired_accuracy,
    )
    fd_test.run()
    