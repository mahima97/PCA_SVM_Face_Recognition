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
from numpy import linalg as LA
from scipy.spatial import distance
# VGG16 standard input shape
dim = (64,48)
# EXPECTED_DIM = (224,224, 3)
# # vgg16 = VGG16(weights='imagenet', include_top=False)
# vgg16 = VGGFace(model='vgg16',include_top=False)
# input = Input(shape=EXPECTED_DIM, name='input')
# output = vgg16(input)
# x = Flatten(name='flatten')(output)
# model = Model(inputs=input, outputs=x)
# print(model.summary())
# def features(img):

# 	# img = image.load_img(img_path, target_size=(224, 224))
# 	img_data = image.img_to_array(img)
# 	img_data = np.expand_dims(img_data, axis=0)
# 	img_data = preprocess_input(img_data)
# 	vgg16_feature = model.predict(img_data)
# 	return vgg16_feature
def shape_to_bb(shape):
#     x_shape = shape.copy()
#     y_shape = shape.copy()
	x_shape = sorted(shape,key = lambda x: x[0])
	y_shape = sorted(shape,key = lambda x: x[1])
	xmin,xmax = x_shape[0][0],x_shape[-1][0]
	ymin,ymax = y_shape[0][1],y_shape[-1][1]
	return xmin,ymin,xmax,ymax

filename = 'svm_weights.pickle'
# detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier('../../pi-face-recognition/haarcascade_frontalface_alt.xml')

predictor = dlib.shape_predictor('../../pi-face-recognition/facial-landmarks/shape_predictor_68_face_landmarks.dat')
# classification_model = pickle.load(open(filename, 'rb'))
EigenFaces = np.load('../models_encodings/EigenFaces.npy')
print('EigenFaces:',EigenFaces.shape)
MeanFace = np.load('../models_encodings/MeanFace.npy')
print('MeanFace:',MeanFace.shape)
# EigenVectors = np.load('EigenVectors.npy')
# print('EigenFaces',EigenFaces.shape)
labels = np.load('../models_encodings/labels.npy')
ProjectionFaces = np.load('../models_encodings/ProjectionFaces.npy')
print('ProjectionFaces',ProjectionFaces.shape)
ProjectionFaces = ProjectionFaces.transpose()
# print(len(labels))
filename = '../models_encodings/svm_weights.pickle'
classification_model = pickle.load(open(filename, 'rb'))

bboxes = []
trackers= []
out_names = []
names = os.listdir('../../pi-face-recognition/Dataset1')
names.sort()
# print(len(names))
cap = cv2.VideoCapture("../../pi-face-recognition/video12.mp4")
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
target_w,target_h = int(width//2),int(height//2)
import time
framenumber = 0
start = time.time()
while True:
	ret, img = cap.read()
	if ret:
		img = cv2.resize(img, (target_w,target_h))
		if framenumber % 1 == 0:
			bboxes = []
			trackers= []
			out_names = []

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# rects = detector(gray,1)
			rects = detector.detectMultiScale(gray)
			face_embb = []
			if len(rects)>0:
				for rect in rects:
					x, y, w, h = rect
					shape_box = dlib.rectangle(x, y, x+w, y+h)
					shape = predictor(gray, shape_box)
					shape = face_utils.shape_to_np(shape)
					x1,y1,x2,y2 = shape_to_bb(shape)
					face_crop = gray[y1:y2,x1:x2]
					face_crop = cv2.resize(face_crop, dim, interpolation = cv2.INTER_AREA)
					# face_embb.append(features(face_crop)[0])

					face_reshape = face_crop.flatten()
					test_mean_sub = face_reshape.reshape((dim[0]*dim[1],1))-MeanFace
					test_projection = np.matmul(EigenFaces.transpose(),test_mean_sub)
					test_projection = test_projection.transpose()
					# print('test_projection:',test_projection[0].shape)
					face_embb.append(test_projection[0])
					# dist= []
					# for p1 in ProjectionFaces:
						# print(test_projection[:10])
						# print(p1[:10])
						# dist.append(distance.euclidean(test_projection,p1)*(10**(-7)))
						# dist.append(distance.cosine(test_projection,p1))
					# dist_2 = LA.norm((test_projection-ProjectionFaces),axis=0)
					# print(len(dist))

					# print(min(dist))
					# if min(dist)<0.001:
					# 	out_names.append(labels[np.argmin(dist)])
					# 	print(labels[np.argmin(dist)],min(dist))
					# else:
					# 	out_names.append('Unknown')
					tracker_dlib = dlib.correlation_tracker()
					rect_dlib = dlib.rectangle(x1, y1, x2, y2)
					tracker_dlib.start_track(img, rect_dlib)
					trackers.append(tracker_dlib)
					bboxes.append([x1,y1,x2,y2])


				face_embb = np.array(face_embb)
				print(face_embb.shape)
				prob_labels = classification_model.predict_proba(face_embb)
				out_names = [names[np.argmax(x)]+":"+str(max(x)) if max(x)>0.65 else 'Unknown' for x in prob_labels]
		else:
			rects_dlib = []
			for i,tracker_dlib in enumerate(trackers):
				tracker_dlib.update(img)
				pos = tracker_dlib.get_position()
				rects_dlib.append((int(pos.left()),int(pos.top()),
									 int(pos.right()), int(pos.bottom())))
			bboxes = [list(elem) for elem in rects_dlib]

		for i, [x1,y1,x2,y2] in enumerate(bboxes):
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			# print(out_names[i])
			cv2.putText(img,out_names[i], (x1, y1 - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
		cv2.imshow("Image", img)
		framenumber = framenumber + 1

		key = cv2.waitKey(1) & 0xFF  # int(1000/fps) is normal speed since waitkey is in ms
		if key == ord("q"):
			break
		# cv2.waitKey(0)
	else:
		break
cap.release()
cv2.destroyAllWindows()
final = time.time() - start
print("[INFO] approx. FPS: {:.2f}".format(framenumber/final))
