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

dim = (64,48)

def shape_to_bb(shape):
#     x_shape = shape.copy()
#     y_shape = shape.copy()
    x_shape = sorted(shape,key = lambda x: x[0])
    y_shape = sorted(shape,key = lambda x: x[1])
    xmin,xmax = x_shape[0][0],x_shape[-1][0]
    ymin,ymax = y_shape[0][1],y_shape[-1][1]
    return xmin,ymin,xmax,ymax


predictor = dlib.shape_predictor('../models_encodings/shape_predictor_68_face_landmarks.dat')


data = []
labels = []
detector = cv2.CascadeClassifier('../models_encodings/haarcascade_frontalface_alt.xml')
names = os.listdir('../dataset')
names.sort()
print(len(names))
encodings = []
print("[INFO] extracting features...")
for i,person in enumerate(names):
    print(i,person)
    if type(person) is int:
        break
    for imagePath in  list(paths.list_images('../dataset/'+person)):
        print(imagePath)
        face = cv2.imread(imagePath)
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(face_gray)

        #If training image contains exactly one face
        if len(rects) >= 1:
            if len(rects) >1:
                # face_box  = max(rects, key=lambda rectangle: (
                #     (rectangle.right()-rectangle.left()) * (rectangle.bottom()-rectangle.top())))
                face_box  = max(rects, key=lambda rectangle: (rectangle[2] *rectangle[3]))
#                 (x, y, w, h) = face_box
            else:
#                 (x, y, w, h) = rects[0]
                face_box = rects[0]
            # face_enc = face_recognition.face_encodings(face)[0]
            # FORMAT for landmark Detection Left, Top,Right,Bottom
            [x,y,w,h] = face_box
            # shape_box = [(x,y),(x+w,y+h)]
            shape_box = dlib.rectangle(x, y, x+w, y+h)
            shape = predictor(face_gray, shape_box)
            shape = face_utils.shape_to_np(shape)
#             print(shape)
            x1,y1,x2,y2 = shape_to_bb(shape)
#             cv2.rectangle(face, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.imshow('img',face)
#             cv2.waitKey(0)
            face_crop = face_gray[y1:y2,x1:x2]
            face_crop = cv2.resize(face_crop, dim, interpolation = cv2.INTER_AREA)
            # print(face_crop.shape)
            face_embb = face_crop.flatten()

            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_embb)
            # print(encodings)
            # labels.append(i)
            labels.append(person)

        else:
            print(person + "/" + imagePath + " was skipped and can't be used for training")


faces = np.array(encodings)
print(faces.shape)
faces = faces.transpose()
np.save('../models_encodings/faces.npy',faces)
print(faces.shape)
np.save('../models_encodings/labels.npy',np.array(labels))
