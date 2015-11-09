from facerec import ImageIO
from facerec.FaceDetector import FaceDetector

import time

def getAccuracy(face_detector, img_data):
    num_of_matches = 0.0
        
    for image in img_data:
        predicted_label = face_detector.classify(image)
            
        if image[1] == predicted_label:
            num_of_matches = num_of_matches + 1
    
    accuracy = num_of_matches / len(img_data)
    
    return (accuracy, num_of_matches)

if __name__ == '__main__': 
    # Start training
    
    directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\train\\'

    face_img_data = ImageIO.loadFaceDetectionImages(directory + 'face', 1)
    non_face_img_data = ImageIO.loadFaceDetectionImages(directory + 'non-face', 0)
    
    runtime = time.time()
    
    # Start training
    
    iterations = 7
    img_size = (19, 19)
    
    face_detector = FaceDetector(iterations, img_size)
    face_detector.train(face_img_data, non_face_img_data)
    
    print 'Training time: {:.2f}'.format(time.time() - runtime)
    
    # Start tweaking threshold on validation set
    
    dev_directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\dev\\face'
    
    img_data = ImageIO.loadFaceDetectionImages(dev_directory, 1)
    
    model = face_detector.getModel()
    initial_threshold = model.getThreshold()
    threshold = initial_threshold
    
    accuracy = getAccuracy(face_detector, img_data)[0]
        
    desired_accuracy = 0.85
    
    runtime = time.time()
    
    while accuracy < desired_accuracy:
        accuracy = getAccuracy(face_detector, img_data)[0]
        
        if accuracy < desired_accuracy:
            threshold = threshold - 0.01;
            model.setThreshold(threshold)
    
    print 'Validation time: {:.2f} (Threshold reduced from {:.2f} to {:.2f})'.format(time.time() - runtime, initial_threshold, threshold)
    
    # Start testing (faces)
    
    runtime = time.time()
    
    test_directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\test\\face'
    
    img_data = ImageIO.loadFaceDetectionImages(test_directory, 1)
    
    accuracy = getAccuracy(face_detector, img_data)
        
    print 'Accuracy (faces): ' + str(accuracy[0] * 100) + ' (' + str(int(accuracy[1])) + '/' + str(len(img_data)) + ')'
    
    print 'Testing time (faces): {:.2f}'.format(time.time() - runtime)
    
    # Start testing (non-faces)
    
    runtime = time.time()
    
    test_directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\test\\non-face'
    
    img_data = ImageIO.loadFaceDetectionImages(test_directory, 0)
    
    accuracy = getAccuracy(face_detector, img_data)
        
    print 'Accuracy (non-faces): ' + str(accuracy[0] * 100) + ' (' + str(int(accuracy[1])) + '/' + str(len(img_data)) + ')'
    
    print 'Testing time (non-faces): {:.2f}'.format(time.time() - runtime)
