'''
argv[1] : model/path/*.h5
'''
from read_data import get_train_data, get_test_data, get_sample_data
import random
import cv2, numpy as np
import pickle
import h5py
import sys

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import metrics

def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def getTestData(chunk,nb_classes,img_rows,img_cols):
	X_test,Y_test = get_test_data(chunk,img_rows,img_cols)
	if (X_test!=None and Y_test!=None):
		X_test/=255
		Y_test = list(map(int, Y_test))
		Y_test=np_utils.to_categorical(Y_test,nb_classes)
	return (X_test,Y_test)

def evaluate(model, nb_epoch, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size):

	accuracy = 0.0
	keys=spatial_test_data.keys()
	random.shuffle(keys)
	cvscores = []
	for chunk in chunks(keys,chunk_size):
		print "Get Test Data: " + str(len(cvscores))
		X_chunk,Y_chunk=getTestData(chunk,nb_classes,img_rows,img_cols)
		if (X_chunk!=None and Y_chunk!=None):
			scores = model.evaluate(X_chunk, Y_chunk, batch_size=512, verbose=1)
			print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
			cvscores.append(scores[1] * 100)

	print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

def VGG_16(img_rows,img_cols,weights_path=None):

	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.9))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.8))
	model.add(Dense(14, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)
	return model

if __name__ == "__main__":

	nb_epoch = 50
	batch_size = 64
	nb_classes = 14
	chunk_size = 128
	img_rows = 224
	img_cols = 224
	model =[]
	print 'Making model...'
	model = VGG_16(img_rows,img_cols, sys.argv[1])

	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

	print 'Compiling model...'
	model.compile(loss='mean_squared_error',
			  optimizer='sgd',
			  metrics=['accuracy'])

	print 'Loading test dictionary...'
	with open('spatial_test_data_new.pickle','rb') as f1:
		spatial_test_data=pickle.load(f1)

	print 'Testing model...'
	evaluate(model, nb_epoch, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size)