# -*- coding: utf-8 -*-
"""
[eigenfaces using PCA]
detector - SSD/ DLIB
"""

# %%
"Loading packages"

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from sklearn.svm import SVC
from pyimagesearch.faces import load_face_dataset,load_face_dlib
from imutils import build_montages
import numpy as np
import imutils
import time
import cv2
import os

# %%
" Command line arguments"
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
ap.add_argument("-n", "--num-components", type=int, default=150,
	help="# of principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())
"""
# %% LOAD PATHS AND INITIALIZE SOME STUFF
print("[INFO] loading dataset...")
input_path = (...)

#%% INITIALIZE DETECTOR MODEL

" Load face detection model from disk (both types, HOG and SSD)"

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

# initialize path for face detection method 1 (SSD)
face_detect_path = (r"...\face_detector")
prototxtPath = os.path.sep.join([face_detect_path, "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([face_detect_path,
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# face detection method 2 ( dlib)
x = -2 # -1 for testing datasets
size = [200,200] #usually don't use
detection_method = 'hog' #hog / cnn

#%% FACE DETECTION METHOD SELECTION 
# can be 'ssd' / 'dlib'
detect_select = 'ssd'

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


# %%
"Load dataset"

# load the training dataset
print("[INFO] Get detected faces and labels for training...")

faces, labels = detect_faces(detect_select, input_path, x=x)
print("[INFO] {} Detected faces in training dataset".format(len(faces)))

# flatten all 2D faces into a 1D list of pixel intensities
# do this to prepare it for the PCA step
pcaFaces = np.array([f.flatten() for f in faces])

# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.20,
	stratify=labels, random_state=420)
(origTrain, origTest, trainX, testX, trainY, testY) = split

"""
Comment : in this segment origTrain & origTest is in the shape of the extracted ROI faces
- train x and test x are the 2D faces flatten to 1D np array
- train y and test y is basically just the labels
"""

#%% Choosing the right n_component value (99% - 305, 95% - 139)
"""
# trying ""to pick the right n_components 
pca = PCA(n_components= None)
trainX = pca.fit_transform(trainX, trainY)

# Create array of explained variance ratios
pca_var_ratios = pca.explained_variance_ratio_

def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
            
    # Return the number of components
    return n_components

n_component = select_n_components(pca_var_ratios, 0.99)
"""

#%%
" Feature extraction - PCA"

# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")

num_components = 0.99

pca = PCA(
	svd_solver="full",
	n_components=num_components,
	whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX) # transform the shape to the num_components, basically reducing it
end = time.time()

# get avg time to compute eigenfaces 
fe_time = (end-start)/len(trainX)
total_fe = (end-start)
print("[INFO] avg time computing eigenfaces per img took {:.4f} seconds".format(total_fe))
 
# %%
" Classifier - SVM"

# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
start = time.time()
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)
end = time.time()
class_time = (end-start)
print("[INFO] training classifier took {:.4f} seconds".format(
	end - start))

total_time = (total_fe + class_time)/len(trainY)
print("[INFO] total feature extractiong and training classifier per img took {:.4f} seconds".format(
	total_time))

# evaluate the model
print("[INFO] evaluating model...")
start = time.time()
predictions = model.predict(pca.transform(testX))
end = time.time()
test_shape = np.shape(testX)
pred_speed = (end-start)/len(testX)
print("[INFO] average recognition per img took {:.4f} seconds".format(
	pred_speed))
print(classification_report(testY, predictions,
	target_names=le.classes_))

# get recognition rate 
# measure recognition rate ( total correct/ total tested)
correct = 0
total = len(testY) #the entire test dataset 

for x in range(len(predictions)):
    if predictions[x] == testY[x]:
        correct+=1
rec_rate = correct/total
print("Recognition Rate:", rec_rate*100)


# %%
"Visualization of recognition/prediction"
# generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
# loop over a sample of the testing data
for i in idxs:
	# grab the predicted name and actual name
	predName = le.inverse_transform([predictions[i]])[0]
	actualName = le.classes_[testY[i]]
	# grab the face image and resize it such that we can easily see
	# it on our screen
	face = np.dstack([origTest[i]] * 3)
	face = imutils.resize(face, width=250)
	# draw the predicted name and actual name on the image
	cv2.putText(face, "pred: {}".format(predName), (5, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	cv2.putText(face, "actual: {}".format(actualName), (5, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	# display the predicted name  and actual name
	print("[INFO] prediction: {}, actual: {}".format(
		predName, actualName))
	# display the current face to our screen
	cv2.imshow("Face", face)
	cv2.waitKey(0)
