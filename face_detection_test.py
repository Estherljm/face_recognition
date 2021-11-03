# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:31:59 2021

[ GET ACCURACY OF FACE DETECTORS - SSD & HOG]

@author: esthe
"""

from sklearn.preprocessing import LabelEncoder
from pyimagesearch.faces import load_face_dataset, load_face_dlib, detect_best_face
import numpy as np
#import argparse
import imutils
import time
import cv2
import os
from imutils import paths
import face_recognition
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#%% LOADING DATASET

# load the testing dataset
print("[INFO] loading testing dataset...")
inputPath = (r"C:\FR_Assignment\face_detect_db")


#%% INITIALIZE DETECTOR MODEL

" Load face detection model from disk (both types, HOG and SSD)"

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

# initialize path for face detection method 1 (SSD)
face_detect_path = (r"C:\FR_Assignment\face_detector")
prototxtPath = os.path.sep.join([face_detect_path, "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([face_detect_path,
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# face detection method 2 ( dlib)
size = [200,200] #usually don't use
detection_method = 'hog'

x = -2

#%% FACE DETECTION METHOD 1 (dlib - HOG)

def detect_dlib(inputPath, detection_method, x):
    imagePaths = list(paths.list_images(inputPath))
    actual_label = []
    pred_label = []
    det_time = 0
  
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        label = imagePath.split(os.path.sep)[x]
        actual_label.append(label)
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

        if len(boxes) >= 1:
            pred_label.append('face')
            
        else:
            pred_label.append('no face')
 
    # calculate detection accuracy  
    matrix = confusion_matrix(actual_label, pred_label, labels=['face','no face'])
    report = classification_report( actual_label, pred_label, labels=['face','no face'])
        
    # print average detection time 
    avg_det_time = det_time/len(imagePaths)
    print("[INFO] Average Detection took {:.4f} seconds".format(avg_det_time))
    # return a 2-tuple of the faces and labels
    return (matrix, report)




#%% FACE DETECTION METHOD 2 (SSD)

def detect_ssd(inputPath, net, minConfidence, x):
    # grab the paths to all images in our input directory, extract
    # the name of the person (i.e., class label) from the directory
    # structure, and count the number of example images we have per
    # face
    imagePaths = list(paths.list_images(inputPath))
    actual_label = []
    pred_label = []
    det_time = 0
    faces = []
    # ori_num = np.shape(imagePaths) #this was done just to check ori num of images from ds
    
    # loop over the image paths
    for imagePath in imagePaths:
        
        label = imagePath.split(os.path.sep)[x]
        actual_label.append(label)
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        image = cv2.imread(imagePath)
        
        # this part is to resize the image by 50%
        scale_percent = 50 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        
        # start timer for detection
        start = time.time()
        # perform face detection
        boxes = detect_best_face(net, image, minConfidence)
        end = time.time()
        det_time = det_time + (end-start)
        faces.append(boxes)
        
        if len(boxes) >= 1:
            pred_label.append('face')
            
        else:
            pred_label.append('no face')


    
    # calculate detection accuracy  
    matrix = confusion_matrix(actual_label, pred_label, labels=['face','no face'])
    report = classification_report( actual_label, pred_label, labels=['face','no face']) 
    
    # detection rate 
    # det_acc = len(boxes)/len(imagePaths)*100
    # print("[INFO] Detection Rate {:.2f}%".format(det_acc))
    
    # print average detection time 
    avg_det_time = det_time/len(imagePaths)
    print("[INFO] Average Detection took {:.4f} seconds".format(avg_det_time))
    # return a 2-tuple of the faces and labels
    return (matrix, report)


#%% EXECUTE FACE DETECTION TEST 

matrix, rep = detect_dlib(inputPath, detection_method, x)

#matrix, rep = detect_ssd(inputPath, net, minConfidence=0.90, x=x)

print('Classification report : \n',rep)

