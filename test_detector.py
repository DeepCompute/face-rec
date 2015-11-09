from facerec import ImageIO
from facerec.FaceDetector import FaceDetector

import time

if __name__ == '__main__': 
    # Start training
    
    directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\face.train.tar\\face.train\\train\\'

    face_img_data = ImageIO.loadFaceDetectionImages(directory + 'face', 1)
    non_face_img_data = ImageIO.loadFaceDetectionImages(directory + 'non-face', 0)
    
    runtime = time.time()
    
    # Start testing
    
    iterations = 7
    img_size = (19, 19)
    
    face_detector = FaceDetector(iterations, img_size)
    face_detector.train(face_img_data, non_face_img_data)
    
    print 'Training time: {:.2f}'.format(time.time() - runtime)
    
    runtime = time.time()
    
    test_directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\test\\face'
    
    img_data = ImageIO.loadFaceDetectionImages(test_directory, 1)
    
    num_of_matches = 0.0
        
    for image in img_data:
        predicted_label = face_detector.classify(image)
            
        if image[1] == predicted_label:
            num_of_matches = num_of_matches + 1
    
    # Print results
        
    accuracy = num_of_matches / len(img_data)
        
    print 'Accuracy: ' + str(accuracy * 100) + ' (' + str(int(num_of_matches)) + '/' + str(len(img_data)) + ')'
    
    print 'Testing time: {:.2f}'.format(time.time() - runtime)
    
