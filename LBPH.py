# -*- coding: utf-8 -*-
"""


@author: esthe
"""

# %%

" import the necessary packages"
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.faces import load_face_dataset, load_face_dlib
from matplotlib import pyplot as plt
import numpy as np
#import argparse
import imutils
import time
import cv2
import os

#%% 
" Valid only when using cmd "

"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input directory of images")
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
"""

#%%
" Load face detection model from disk (both types, HOG and SSD)"

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

# initialize path for face detection method 1
face_detect_path = (r"C:\FR_Assignment\face_detector")
prototxtPath = os.path.sep.join([face_detect_path, "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([face_detect_path,
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# face detection method 2 ( dlib)
xtrain = -2
size = [200,200] #usually don't use
detection_method = 'hog'

#%%
"Load whole dataset of detected faces with labels"
# load the faces dataset
print("[INFO] loading dataset...")
db_path = (r"C:\FR_Assignment\DB5_PCA_FULL")

# face detection method 1 (ssd) 
# (faces, labels) = load_face_dataset(db_path, net, minConfidence=0.90, minSamples=10, x = xtrain)
# print("[INFO] {} images in dataset".format(len(faces)))

# face detection method 2 (hog/cnn)
(faces, labels) = load_face_dlib(db_path, size,detection_method, xtrain)
print("[INFO] {} Detected faces in training dataset".format(len(faces)))

# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#%% Loop recognition multiple times with different random seeds to get avg 

loop_count = 10 
train_time = 0
pred_time = 0
t_rec_rate = 0

for i in range(loop_count):
    
    # construct our training and testing split
    (trainX, testX, trainY, testY) = train_test_split(faces,
        labels, test_size=0.25, stratify=labels)
    
    # training 
    recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=7, grid_x=7, grid_y=7)
    start = time.time()
    recognizer.train(trainX, trainY)
    end = time.time()
    r_time = (end-start)/len(trainX) #train time per image in trainX
    train_time = train_time + r_time # total avg training time per image 
    #print("[INFO] training {0}. took {:.4f} seconds".format(i,train_time))
    
    # predictions 
    predictions = []
    confidence = []
    start = time.time()
    # loop over the test data
    for i in range(0, len(testX)):
        # classify the face and update the list of predictions and
        # confidence scores
        (prediction, conf) = recognizer.predict(testX[i])
        predictions.append(prediction)
        confidence.append(conf)
    # measure how long making predictions took
    end = time.time()
    pred_time = pred_time + (end-start)

    # measure recognition rate ( total correct/ total tested)
    correct = 0
    total = len(testY) #the entire test dataset 
    
    for x in range(len(predictions)):
        if predictions[x] == testY[x]:
            correct+=1
    rec_rate = correct/total
    t_rec_rate = t_rec_rate + rec_rate
    print("Recognition Rate:", rec_rate*100)    

# count average time taken 
avg_train_time = round(train_time/loop_count,4)
print("[INFO] avg training per image for {0} loops took {1} seconds".format(loop_count,avg_train_time))
avg_pred_time = round((pred_time/len(testX))/loop_count,4)
print("[INFO] avg recognition per image of {0} loops took {1} seconds".format(loop_count,avg_pred_time))
print("[INFO] avg recognition rate of {0} loops is {1}% ".format(loop_count,round(t_rec_rate/loop_count,4)*100))




# %%
"train our LBP face recognizer"

#print("[INFO] training face recognizer...")

#print("[INFO] training took {:.4f} seconds".format(end - start))

# initialize the list of predictions and confidence scores
#print("[INFO] gathering predictions...")

#print("[INFO] inference took {:.4f} seconds".format(end - start))
# show the classification report
print(classification_report(testY, predictions,
    target_names=le.classes_))

#%% Visualization of some samples 
# generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
 
# loop over a sample of the testing data
for i in idxs:
    # grab the predicted name and actual name
    predName = le.inverse_transform([predictions[i]])[0]
    actualName = le.classes_[testY[i]]
    # grab the face image and resize it such that we can easily see
    # it on our screen
    face = np.dstack([testX[i]] * 3)
    face = imutils.resize(face, width=250)
    # draw the predicted name and actual name on the image
    cv2.putText(face, "pred: {}".format(predName), (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(face, "actual: {}".format(actualName), (5, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # display the predicted name, actual name, and confidence of the
    # prediction (i.e., chi-squared distance; the *lower* the distance
    # is the *more confident* the prediction is)
    print("actual")
    print("[INFO] prediction: {}, actual: {}, confidence: {:.2f}".format(
        predName, actualName, confidence[i]))
    # display the current face to our screen
    cv2.imshow("Face", face)
    cv2.waitKey(0)




#%%
"""
# Visualize LBP images
def get_pixel(img, center, x, y):
      
    new_value = 0
      
    try:
        # If local neighbourhood pixel 
        # value is greater than or equal
        # to center pixel values then 
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
              
    except:
        # Exception is required when 
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass
      
    return new_value
   
# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
   
    center = img[x][y]
   
    val_ar = []
      
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
      
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
      
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))
      
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
      
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
      
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
      
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))
      
    # left
    val_ar.append(get_pixel(img, center, x, y-1))
       
    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
   
    val = 0
      
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
          
    return val

path = (r"C:\FR_Assignment\test_db\abu.jpg")
img_bgr = cv2.imread(path)

# got dash because input img is rgb, 3 channels, if grayscale then remove _    
height, width, _ = img_bgr.shape
   
# We need to convert RGB image 
# into gray one because gray 
# image has one channel only.
img_gray = cv2.cvtColor(img_bgr,
                        cv2.COLOR_BGR2GRAY)
   
# Create a numpy array as 
# the same height and width 
# of RGB image
img_lbp = np.zeros((height, width),
                   np.uint8)
   
for i in range(0, height):
    for j in range(0, width):
        img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
  
plt.imshow(img_bgr)
plt.show()
   
plt.imshow(img_lbp, cmap ="gray")
plt.show()

"""