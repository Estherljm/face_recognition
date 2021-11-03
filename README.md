# Comparison between classical and deep learning face recognition techniques 
The purpose of this project is to identify the best face recognition technique for the application of an attendance system in Malaysia. Hence, the best model will be chosen according to these criterias:

1. Recognition Rate 
2. Recognition Speed 
3. Number of sample images needed per person (The lesser the better)

There were several datasets involved in this project, the full dataset - 20 sample images per person, partial dataset - 5 sample images per person. 
The reason behind this is to test the performance of the models on different dataset size to identify the best one that can perform well with the partial dataset for practicality reasons for an attendance system.

# Workflow
![alt text](https://github.com/Estherljm/face_recognition/blob/main/Images/Picture1.png)

The input images are loaded from either the training or testing dataset. For each image, face detection will be performed with either HOG or SSD. 
The detectors will then output a set of coordinates which denotes the position of the bounding box around the face. 
Next, the detected face will undergo some pre-processing to meet the conditions required for each model depending on the type of feature extraction used. 
The processed faces are then fed into the feature extraction algorithms to extract distinctive features which creates a unique face representation of each input. 
If the image was from the training dataset, the face representation will be saved into the database. 
Else if the image was from a testing dataset, the face representation will be compared to those in the existing database to find the best matched face using either SVM or Euclidean distance respective to the model. 
The result of the output will be the predicted name of the face.

# Results 
There are 3 main parts that was discussed to meet the objective of this project:


1. Speed comparison between face recognition models 
2. Recognition rate comparison between face recognition models 
3. Recognition rate comparison between face recognition models by race

## 1.1 Speed comparison between face recognition models 
### Average Detection Speed
![alt text](https://github.com/Estherljm/face_recognition/blob/main/Images/detector_speed.JPG)
- SSD detector is 3 times faster than the HOG detector
- The SSD detector is faster than HOG as it only examines the image once to locate the position of the face
- HOG detection involves a longer feature extraction process and classification time as it is performed in patches 

### Average Feature Extraction Speed 
![alt text](https://github.com/Estherljm/face_recognition/blob/main/Images/fe_speed.JPG)
- Average time taken was also notably longer for deep learning-based methods such as FaceNet and VGGFace in comparison to the classical methods
- When the comparison is narrowed down between VGGFace and FaceNet, the VGGFace takes 3 times longer than FaceNet 

### Average Recognition Speed
![alt text](https://github.com/Estherljm/face_recognition/blob/main/Images/rec_speed.JPG)
- Models using the SSD as an obvious advantage as it is faster than the HOG detector
- Model using VGGFace + HOG had the slowest recognition speed at an average of 0.2452 seconds per image
- Model using PCA + SSD had the fastest recognition speed at an average of 0.0336 seconds per image

## 1.2 Recognition rate comparison between face recognition models 
![alt text](https://github.com/Estherljm/face_recognition/blob/main/Images/rec_rate.JPG)
- Models using VGGFace and FaceNet achieved at least above 80% recognition rate while the PCA and LBPH suffered poorly with less than 45% recognition rate
- Classical feature extraction methods such as PCA and LBPH are highly influenced on the conditions of the image
- Performance of models using FaceNet were least affected by size of dataset and achieved an average of 97-98% recognition rate 
- Models using VGGFace has lesser generalization ability compared to FaceNet as it was more affected by variations in facial expression and pose and similarity in makeup and facial hair 

## 1.3 Recognition Rate by Racial Category
![alt text](https://github.com/Estherljm/face_recognition/blob/main/Images/rec_rate_race.JPG)
- Most of the face recognition models developed can recognize Malay, Indian and Chinese faces with at least 90% recognition rate
- No obvious racial bias in performance of the models
- Difference in performance of models by race was most likely affected by the condition of the images rather than the skin colour of the individuals

# Final best face recognition model 
SSD + FaceNet

Recognition rate - 97.95% 

Speed - 0.0812 seconds 

No. of sample images per person - can work well with 5 images each 
