#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
import pickle
# from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
from sklearn.svm import SVC
import face_recognition
import os
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# from keras.layers import Input, Flatten
# from keras.preprocessing import image
import numpy as np
# from keras.applications.vgg16 import preprocess_input
# from keras_vggface.vggface import VGGFace
import dlib
from imutils import face_utils
# VGG16 standard input shape
from numpy import linalg as LA

# In[2]:

dim = (64,48)

labels = np.load('../models_encodings/labels.npy')
faces = np.load('../models_encodings/faces.npy')
mean_face = np.mean(faces,axis=1)
mean_face = mean_face.reshape((dim[0]*dim[1],1))
print(mean_face.shape)

np.save('../models_encodings/MeanFace.npy',mean_face)
mean_sub_faces = faces-mean_face
print(mean_sub_faces.shape)

cov_matrix = np.matmul(mean_sub_faces.transpose(),mean_sub_faces)
print(cov_matrix.shape)

EigValues,EigVectors = LA.eig(cov_matrix)
print(EigVectors.shape,EigValues.shape)
EigenVectors = []
for col in range(len(EigValues)):
    if EigValues[col] > 150000:
        EigenVectors.append(EigVectors[:,col])
EigenVectors = np.array(EigenVectors)
EigenVectors = EigenVectors.transpose()
# print(EigenVectors.shape)

EigenFaces = np.matmul(mean_sub_faces,EigenVectors)
print(EigenFaces.shape)
ProjectionFaces = np.matmul(EigenFaces.transpose(),mean_sub_faces)
ProjectionFaces = ProjectionFaces.transpose()
print(ProjectionFaces.shape)
np.save('../models_encodings/EigenVectors.npy',EigenVectors)
np.save('../models_encodings/EigenFaces.npy',EigenFaces)
np.save('../models_encodings/ProjectionFaces.npy',ProjectionFaces)

# encodings = np.array(encodings)
# # print(labels[567])
clf = SVC(kernel='linear', probability=True, random_state=0)
clf.fit(ProjectionFaces, labels)
filename = 'PCA_face_features.pickle'
pickle.dump(clf, open('../models_encodings/svm_weights.pickle', 'wb'))
