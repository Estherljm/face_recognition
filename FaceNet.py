# -*- coding: utf-8 -*-
"""
[ Facenet encoding faces ] (expects 160,160)
"""

# %%

" import the necessary packages"
from pyimagesearch.faces import load_face_dataset, load_face_dlib,findCosineDistance,findEuclideanDistance #custom functions
import numpy as np
import imutils
import time
import cv2
import os
from imutils import paths
from face_analysis_tensorflow2.architecture import InceptionResNetV2
import pickle

#%% LOADING TRAINING SET AND PICKLE FILE LOCATION
# load the training dataset
print("[INFO] loading training dataset...")
train_path = (...)

# load the testing dataset
print("[INFO] loading testing dataset...")
test_path = (...)

# path to store pickle file of encoded faces
pickle_path = (r"C:\FR_Assignment\DB5_SSD_FaceNet_encoding50")

#%% INITIALIZE DETECTOR MODEL

" Load face detection model from disk (both types, HOG and SSD)"

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

# initialize path for face detection method 1 (SSD)
face_detect_path = (r"C:\...\face_detector")
prototxtPath = os.path.sep.join([face_detect_path, "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([face_detect_path,
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# face detection method 2 ( dlib)
x= -2
size = [200,200] #usually don't use
detection_method = 'hog'

#%% FACE DETECTION METHOD SELECTION 
# can be 'ssd' / 'dlib'
detect_select = 'dlib'

def detect_faces(detect_select, path, x):
    
    if detect_select == 'ssd':
        # face detection method 1 (ssd) 
        (faces, labels) = load_face_dataset(path, net, minConfidence=0.90, minSamples=10, x=x)
        print("[INFO] {} Detected faces in this dataset".format(len(faces)))
        
    elif detect_select == 'dlib':
        # face detection method 2 (hog/cnn)
        (faces, labels) = load_face_dlib(path, size,detection_method, x)
        print("[INFO] {} Detected faces in this dataset".format(len(faces)))

    return faces, labels

#%%
"""
PHASE 1 : DETECT AND ENCODE FACES IN TRAINING DATASET
"""

#%% DETECTING FACES AND LABELS IN TRAINING SET 
print("[INFO] Detecting faces in training dataset...")
# get faces and labels detected from training images
faces, labels = detect_faces(detect_select, train_path, x=x)

#%% BUILD FACENET NETWORK FOR ENCODING
"Encode faces in training set"

"Load encoder"
# load network architecture of encoder (facenet)
face_encoder = InceptionResNetV2()
# path of pretrained weights
# if cannot run this part means the h5 file has been corrupted 
path = (r"C:\...\facenet_keras.h5") 
# load weights into network 
face_encoder.load_weights(path)


# function to normalize images
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

# function to encode using facenet
def facenet_encode(faces,labels):
    
    # initialize the list of known encodings and known names
    knownEncodings = [] # store face encodings
    knownNames = [] # store labels/names of each encoding 
    
    start = time.time()
    for i in range(len(faces)):
        
        print("[INFO] processing image {}/{}".format(i + 1,
            len(faces)))
        name = labels[i]
        face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2RGB)
        face = normalize(face)
        face = cv2.resize(face, (160,160))
        face_d = np.expand_dims(face, axis=0)
        
        encodings = face_encoder.predict(face_d)[0]
        
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encodings)
        knownNames.append(name)
    
    
    end = time.time()
    print("[INFO] Encoding took {:.4f} seconds".format((end - start)/len(knownEncodings)))

    return knownEncodings, knownNames

#%% GET ENCODINGS OF TRAINING SET 
# get encoded images using facenet
knownEncodings, knownNames = facenet_encode(faces, labels)

data = {"encodings": knownEncodings, "names": knownNames} 

#%% DUMP ENCODED TRAINING DATA INTO PICKLE FILE 
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
f = open(pickle_path, "wb") #set path to store encoding HERE
f.write(pickle.dumps(data))
f.close()

#%% 
"""
PHASE 2 (RECOGNITION PHASE) on testing dataset

"""
#%% GET COUNT OF TOTAL IMAGES IN TEST 

# load the testing dataset
print("[INFO] loading testing dataset...")
test_path = (...)

# (this part is for recognition rate calculation)

test_path_total = list(paths.list_images(test_path)) 
#%% LOADING ENCODINGS OF TRAINING DATA 

# ssd
train_encoding_path = (r"...\DBFULL_front_SSD_FaceNet_encoding50")


#train_encoding_path = (r"...\DB5_SSD_FaceNet_encoding50")
#train_encoding_path =(r"...\DBFULL_front_HOG_FaceNet_encoding50")

# load the known faces and embeddings
print("[INFO] loading encodings...") 
data = pickle.loads(open(train_encoding_path, "rb").read()) 

#%% GET ENCODINGS OF TEST IMAGES 
x_test = -2
# get faces and labels detected from training images
testfaces, testlabels = detect_faces(detect_select, test_path,x_test)

# get encoded images using vggface
test_knownEncodings, test_knownNames = facenet_encode(testfaces, testlabels)


#%% LOAD AND RECOGNIZE TEST DATA 
# clas_choice refers to the classification method 
# got 2 options 'ed', 'cos'
class_choice = 'ed'

def facenet_rec_test(detect_select,test_path, data, class_choice):
    
    #initialize matches
    pred_test_name = []
    rec_time = 0
    
    for i in range(len(test_knownNames)):
        test_encoding = test_knownEncodings[i]
        #test_name = test_knownNames [i]
        # classification method 1 (euclidean dist)
        if class_choice == 'ed':
            best_ed = 200
            start = time.time()
            for x in range(len(data["names"])):
                ed = findEuclideanDistance(test_encoding,data["encodings"][x])
                if ed < best_ed:
                    best_ed = ed
                    match_name = data["names"][x]
            end = time.time()
            rec_time = rec_time + (end-start)
        # classification method 2 (cosine similarity)
        elif class_choice == 'cos':
            best_cos = 200
            start = time.time()
            for x in range(len(data["names"])):
                cosine = findCosineDistance(test_encoding,data["encodings"][x])
                if cosine < best_cos:
                    best_cos = cosine
                    match_name = data["names"][x]
            end = time.time()
            rec_time = rec_time + (end-start)
                    
        pred_test_name.append(match_name)
    print("[INFO] Average recognition time per img took {:.4f} seconds".format(rec_time/len(test_knownNames)))
        
    return pred_test_name

#%% GET PREDICTION/RECOGNITION FROM TEST DATA

pred_test_name = facenet_rec_test(detect_select,test_path, data, class_choice)          
            
# GET RECOGNITION RATE                 

# initialize to count for recognition rate    
correct = 0
total = len(test_path_total) #the entire test dataset           
wrong_list = []  
wrong_count = 0          
            
for i in range(len(test_knownNames)):     
    if pred_test_name[i] == test_knownNames[i]:
        correct+=1
    else:
        wrong_list.append(i)
        wrong_count+=1

rec_rate = correct/total
print("Recognition Rate:", rec_rate*100)             
        

#%% VISUALIZATION
# printing errors/ wrong recognized faces
for i in (wrong_list) :
    # grab the predicted name and actual name
    predName = pred_test_name[i]
    actualName = test_knownNames[i]
    # grab the face image and resize it such that we can easily see
    # it on our screen
    face = np.dstack([testfaces[i]] * 3)
    face = imutils.resize(face, width=250)
    # draw the predicted name and actual name on the image
    cv2.putText(face, "pred: {}".format(predName), (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(face, "actual: {}".format(actualName), (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # display the predicted name, actual name, and confidence of the
    # prediction (i.e., chi-squared distance; the *lower* the distance
    # is the *more confident* the prediction is)
    print("actual")
    print("[INFO] prediction: {}, actual: {}".format(
        predName, actualName))
    # display the current face to our screen
    cv2.imshow("Face", face)
    cv2.waitKey(0)
    










