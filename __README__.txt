THERE ARE TOTAL 4 FACE RECOGNITION MODELS IN THIS FILE 

*****************************************MODELS******************************************
1. EIGENFACES.PY
2. LBPH.PY
3. FACENET_ENCODING.PY 
4. VGGFACE_ENCODING.PY

*****************************************DATASET****************************************** 
1. FR_DB 
- FULL DATASET (USED FOR FACENET, VGGFACE) 
2. FR_DB_PCA 
- FULL DATASET (USED FOR PCA, LBPH) 
3. FACE_DETECT_DB
- THIS IS THE FACE_NOFACE DATASET TO TEST PERF OF DETECTION MODEL IN RESULTS 1
4. FR_CINA, FR_INDIAN, FR_MALAY
- DATASET BY RACE 
5. DB_5
- PARTIAL DATASET (USED FOR FACENET, VGGFACE) 
6. DB5_PCA_FULL
- PARTIAL DATASET (USED FOR PCA, LBPH) 
7. CALTECH_FACES
- EXTERNAL DATASET JUST FOR TESTING 
8. FINAL_TEST_DB 
- TESTING DATASET 

*****************************************ENCODINGS***************************************
THIS IS ONLY FOR THE FACENET AND VGGFACE METHODS WHERE THE ENCODINGS ARE SAVED THEN RELOADED.
- ALL THE PICKLE FILES 

*****************************************MODELS******************************************
PRETRAINED MODEL ARCHITECTURE AND WEIGHTS 
1. tensorflow-101-master 
- INCEPTION_RESTNET_V1 (USED FOR FACENET)

2. face_analysis_tensorflow2
- FACENET_KERAS.H5 (FACENET WEIGHTS) 
- VGG_FACE_WEIGHTS (VGGFACE WEIGHTS) 

3.face_detector (FOR SSD)
- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel

************************************FUNCTIONS*********************************************
1. pyimagesearch.faces 
- FUNCTION TO LOAD AND DETECT DATA 


