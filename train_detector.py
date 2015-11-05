import sys
sys.path.append('src')
import ImageIO
from FaceDetector import FaceDetector

if __name__ == '__main__': 
    directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\face.train.tar\\face.train\\train\\'

    face_img_data = ImageIO.loadFaceDetectionImages(directory + 'face')
    non_face_img_data = ImageIO.loadFaceDetectionImages(directory + 'non-face')
    
    face_detector = FaceDetector(face_img_data, non_face_img_data)
    face_detector.fitModel()
    
    test_directory = 'C:\\Users\\Sanghyun\\Downloads\\faces.tar\\faces\\test\\face'
    
    img_data = ImageIO.loadFaceDetectionImages(test_directory)
    face_detector.classify(img_data)
    
