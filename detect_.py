# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
model=KNeighborsClassifier(n_neighbors=1)
# construct the argument parser and parse the arguments
"""ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())"""

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
traindata=[]
testdata=[]
trainlabels=[]
testlabels=[]
for imagepath in paths.list_images("database"):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imagepath)
    
    make=imagepath.split('/')[-2]
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        b=[]
        
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        
            roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            roi=cv2.resize(roi,(300,200))
            
            hog_roi=feature.hog(roi,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True)
            print(hog_roi.shape) 
            #cv2.imshow("ROI", roi)
            #cv2.imshow("Image", clone)
            b.append(hog_roi)
            #cv2.waitKey(0)
        b=np.array(b)
        print(b.shape)
    print("face")
    traindata.append(b)
    
    trainlabels.append(make)
    #print(traindata)
    print('\n')
traindata=np.array(traindata)
print(traindata.shape)
print("Training.....")
model.fit(traindata,trainlabels)
print("preparing for testing")
for imagepath in paths.list_images("testdata"):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imagepath)
    
    make=imagepath.split('/')[-2]
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        b=[]
        
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        
            roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            roi=cv2.resize(roi,(300,200))
            
            H,hog_roi=feature.hog(roi,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True)
        
            cv2.imshow("ROI", roi)
            cv2.imshow("Image", clone)
            cv2.imwrite('/home/dristibutola/database/"+str(name)+"/"+str(
            b.append(roi)
            cv2.waitKey(0)
            pred=model.predict(H.reshape(1,-1))[0]
            print(pred)
    print("face")
    testdata.append(b)
    testlabels.append(make)
    
"""   output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)"""
