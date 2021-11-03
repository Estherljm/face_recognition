# -*- coding: utf-8 -*-
"""
This part is to load the dataset,and detect the faces in the images 

input - image paths
output - cropped detected faces [type(np arrays), size(300,300)] and 
labels [type - np arrays of strings] (eg. 'Amber Chia')

2 types of detection and loading method 
- load_face_dataset (using ssd)
- load_face_dlib (dlib - HOG/CNN)

@author: esthe
"""
#%%
" Loading packages"
from imutils import paths
import numpy as np
import cv2
import os
import imghdr
from scipy import spatial


# %%
" Detect Faces"

def detect_faces(net, image, minConfidence):
    # grab the dimensions of the image and then contruct a blob from it 
    (h, w) = image.shape[:2] #this is the height & width of the blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network to obtain face detections,
    # initialize a list to store the predicted bounding boxes
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    
    # loop over the detections 
    for i in range(0, detections.shape[2]):
        # extract the condfidence (i.e, prob associated with the detection)
        confidence = detections[0,0,i,2]
        
        # filter out weak detections by ensuring the condfidence is > min condfidence 
        if confidence > minConfidence:
            # computer the (x,y) coordinates of the bounding box for the object 
            box = detections[0,0,i, 3:7]* np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # update our bounding box results list
            boxes.append((startX, startY, endX, endY))
            
    # return the face detection bounding boxes
    return boxes
            
#%% 
" Detect faces but choose 1 best box only"

def detect_best_face(net, image, minConfidence):
    # grab the dimensions of the image and then contruct a blob from it 
    (h, w) = image.shape[:2] #this is the height & width of the blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network to obtain face detections,
    # initialize a list to store the predicted bounding boxes
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    best_boxes = []
    best_confidence = minConfidence
    
    # loop over the detections 
    for i in range(0, detections.shape[2]):
        # extract the condfidence (i.e, prob associated with the detection)
        confidence = detections[0,0,i,2]
        
        # filter out weak detections by ensuring the condfidence is > min condfidence 
        if confidence > best_confidence:
            # computer the (x,y) coordinates of the bounding box for the object 
            box = detections[0,0,i, 3:7]* np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # update our bounding box results list
            boxes.append((startX, startY, endX, endY))
            
            # replace the best boxes
            best_boxes = boxes
            
            # replace the best confidence
            best_confidence = confidence
            
    # return the face detection bounding boxes
    return best_boxes



#%%
" Loading Dataset"

""" 
inputPath - path to our dataset
net - face detection model 
minConfidence - min conf for a postive detection 
minSamples - min num of imgs required for each face/class

process 
- load dataset
- detect faces 
- extract face ROI from detected faces 
- resize to fixed size (required for PCA)
- convert BGR to Grayscale

return 
- faces, labels
"""

# initialy minConfidence=0.5, minSamples=15
# minConfidence = 0.95

#types = ['jpg', 'jpeg']


def load_face_dataset(inputPath, net, minConfidence,
    minSamples, x):
    # grab the paths to all images in our input directory, extract
    # the name of the person (i.e., class label) from the directory
    # structure, and count the number of example images we have per
    # face
    imagePaths = list(paths.list_images(inputPath))
    # names = [p.split(os.path.sep)[x] for p in imagePaths]
    # (names, counts) = np.unique(names, return_counts=True)
    # names = names.tolist()
    # initialize lists to store our extracted faces and associated
    # labels
    faces = []
    labels = []
    det_time = 0
    
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        image = cv2.imread(imagePath)
        
        # this part is to resize the image by 50%
        scale_percent = 50 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        """
        # check whether it is jpeg/jpg
        img_type = imghdr.what(imagePath)
        
        
        # converting to jpg
        if img_type not in types:
            outfile = imagePath.split('.')[0] + '.jpg'
            cv2.imwrite(outfile, image, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
            
            # delete original file 
            os.remove(imagePath)
            
        """
        
        name = imagePath.split(os.path.sep)[x]
        name = os.path.splitext(name)[0]
        # only process images that have a sufficient number of
        # examples belonging to the class
        # if counts[names.index(name)] < minSamples:
        #     continue
        # start timer for detection
        start = time.time()
        # perform face detection
        boxes = detect_best_face(net, image, minConfidence)
        
        end = time.time()
        det_time = det_time + (end-start)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the face ROI, resize it, and convert it to
            # grayscale
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (300, 300)) #what if change to 160,160 (47,62)
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # update our faces and labels lists
            faces.append(faceROI)
            labels.append(name)
    # convert our faces and labels lists to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    # print detected "accuracy" 
    det_acc = len(faces)/len(imagePaths)*100
    print("[INFO] Detection Rate {:.2f}%".format(det_acc))
    # print average detection time 
    avg_det_time = det_time/len(imagePaths)
    print("[INFO] Average Detection took {:.4f} seconds".format(avg_det_time))
    # return a 2-tuple of the faces and labels
    return (faces, labels)

#%% DETECT WITH DLIB 

import face_recognition
import cv2
import time
import numpy as np
from imutils import paths
import os
import re

# face detection method (cnn or hog)
detection_method = 'hog'

# x can be -2 or -1 depending on dataset 
# -2 for training, -1 testing 
# size refers to the size of the resized image 
xtrain = -2

inputPath = (r"C:\FR_Assignment\test_db")
testpath = (r"C:\FR_Assignment\FR_DB_PCA\Yuna\Yuna.jpg")
size = [200,200]

 
def load_face_dlib(inputPath, size, detection_method, x):
    # grab the paths to all images in our input directory, extract
    # the name of the person (i.e., class label) from the directory
    # structure, and count the number of example images we have per
    # face
    imagePaths = list(paths.list_images(inputPath))
    #names = [p.split(os.path.sep)[-2] for p in imagePaths]
    #(names, counts) = np.unique(names, return_counts=True)
    #names = names.tolist()
    # initialize lists to store our extracted faces and associated
    # labels
    faces = []
    labels = []
    # ori_num = np.shape(imagePaths) #this was done just to check ori num of images from ds
    det_time = 0
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        name = imagePath.split(os.path.sep)[x]
        name = os.path.splitext(name)[0]
        # load and process images 
        image = cv2.imread(imagePath)
        # this part is to resize the image by 50%
        scale_percent = 50 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        # start timer for detection
        start = time.time()
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(image,
            model=detection_method) # need to insert detection method HERE    
        end = time.time()
        det_time = det_time + (end-start)
        
        # ensure 1 box/ detection only 
        if len(boxes) >= 1:
            top = boxes[0][0]
            right = boxes[0][1]
            bottom = boxes[0][2]
            left = boxes[0][3]
        
            # extract the face ROI, resize it, and convert it to
            # grayscale
            faceROI = image[top:bottom,left:right]
            faceROI = cv2.resize(faceROI, (300, 300))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # update our faces and labels lists
            faces.append(faceROI)
            labels.append(name)
    # convert our faces and labels lists to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    # print detected "accuracy" 
    det_acc = len(faces)/len(imagePaths)*100
    print("[INFO] Detection Rate {:.2f}%".format(det_acc))
    # print average detection time 
    avg_det_time = det_time/len(imagePaths)
    print("[INFO] Average Detection took {:.4f} seconds".format(avg_det_time))
    # return a 2-tuple of the faces and labels
    return (faces, labels)


#(faces, labels) = load_face_dlib(inputPath, size,detection_method, xtrain)

#%% COSINE SIMILARITY 

"""
# method by https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
def findCosineDistance(train_encoding, test_encoding):
    a = np.matmul(np.transpose(train_encoding), test_encoding)
    b = np.sum(np.multiply(train_encoding, train_encoding))
    c = np.sum(np.multiply(test_encoding, test_encoding))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
"""
# method by https://www.codegrepper.com/code-examples/python/how+to+calculate+cosine+similarity+in+python

def findCosineDistance(train_encoding, test_encoding):
    cosine = 1 - spatial.distance.cosine(train_encoding, test_encoding)

    return cosine


#%% EUCLIDEAN DISTANCE 

def findEuclideanDistance(train_encoding, test_encoding):
    euclidean_distance = train_encoding - test_encoding
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance





#%%
"""
image = cv2.imread(testpath)

# displaying image
cv2.imshow('image',image)
cv2.waitKey(0)

cv2.imshow('image',faceROI)
cv2.waitKey(0)
"""
# %%















