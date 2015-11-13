'''
Tests the face recognition module with data from yalefaces.

'''

from facerec import ImageIO
from facerec.FaceRecognizer import FaceRecognizer

import sys, time


def test_loocv(instances):
    '''
    Tests the classifier with LOOCV.

    Args:
        instances (list<tuple>): List of label/data/mode tuples.
    '''



if __name__ == '__main__':
    '''
    Perform leave-one-out cross-validation on yalefaces dataset.
    '''

    instances = None

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'a':
            print 'Loading Yalefaces (A) dataset'
            instances = ImageIO.loadYalefacesImages('data/yalefaces')
        elif sys.argv[1].lower() == 'b':
            print 'Loading extended Yalefaces (B) dataset'
            instances = ImageIO.loadExtendedCroppedYalefaces(
                    'data/yalefaces-ext')

    if instances is None:
        print 'Please specify which Yalefaces set to use (A or B)'
        sys.exit(1)


    modes = list()
    for instance in instances:
        if instance[2] not in modes:
            modes.append(instance[2])

    # Initialize tallies

    avg_training_time = 0
    avg_classification_time = 0

    # Number of correct classifications overall
    correct_count = 0

    # Number of correct classifications per mode
    total_mode = dict()
    correct_mode = dict()

    for mode in modes:
        total_mode[mode] = 0.0
        correct_mode[mode] = 0.0

    print '\nPer-case results:\n'

    for i in range(0, len(instances)):

        # Take first instance from the list
        test_instance = instances.pop(0)

        label, features, mode = test_instance

        # Train with all other instances
        runtime = time.time()
        face_recognizer = FaceRecognizer()
        face_recognizer.train(instances)
        avg_training_time += (time.time()-runtime)/len(instances)

        # Make a prediction with removed instance
        runtime = time.time()
        predicted_label = face_recognizer.classify(features)
        avg_classification_time += (time.time()-runtime)/len(instances)

        # Increment tallies
        if predicted_label == label:
            correct_count += 1.0
            correct_mode[mode] += 1.0
        total_mode[mode] += 1.0

        print '[Case {:03d}] Predicted {:2d} as {:2d} ({:^11})'.format(
                i, label, predicted_label, mode)

        # Put instance back on the end of the list
        instances.append(test_instance)

    print '\n---- ---- Results ---- ----\n'
    print 'Overall accuracy: {:.2f} ({:3d}/{:3d})'.format(
            correct_count/len(instances), int(correct_count), len(instances))
    print '\nPer mode:'
    for mode in modes:
        print '\t{:>11} accuracy: {:.2f} ({:2d}/{:2d})'.format(
                mode, correct_mode[mode]/total_mode[mode],
                int(correct_mode[mode]), int(total_mode[mode]))
    print '\nRuntimes:'
    print '\tAverage training time:       {:.2f}'.format(avg_training_time)
    print '\tAverage classification time: {:.2f}'.format(
            avg_classification_time)

