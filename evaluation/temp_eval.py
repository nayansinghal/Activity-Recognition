import os
import numpy as np
import h5py
import gc
import temporal_stream_data as tsd
import pickle
import random
import sys

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization

def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def getTestData(chunk,nb_classes,img_rows,img_cols):
	X_train,Y_train=tsd.stackOFTest(chunk,img_rows,img_cols)
	if (X_train!=None and Y_train!=None):
		X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def createModel(img_channels, img_rows, img_cols, nb_classes, weights_path=None):

	model = Sequential()

	model.add(Convolution2D(48, 7, 7, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(96, 5, 5, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(512, 3, 3, border_mode='same'))	
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(512, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	if weights_path:
		model.load_weights(weights_path)
	return model  

def evaluate(model, temporal_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size):

	print 'Testing model...'
	accuracy = 0.0
	keys=temporal_test_data.keys()
	random.shuffle(keys)
	cvscores = []

	for chunk in chunks(keys,chunk_size):
		print "Get Test Data: " + str(len(cvscores))
		X_chunk,Y_chunk=getTestData(chunk,nb_classes,img_rows,img_cols)
		if (X_chunk!=None and Y_chunk!=None):
			scores = model.evaluate(X_chunk, Y_chunk, batch_size=512, verbose=1)
			print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
			cvscores.append(scores[1] * 100)

def CNN():
	input_frames=10
	batch_size=64
	nb_classes = 14
	img_rows, img_cols = 150,150
	img_channels = 2*input_frames
	chunk_size=128

	print 'Preparing architecture...'
	print sys.argv[1]
	model = createModel(img_channels, img_rows, img_cols, nb_classes, sys.argv[1])

	print 'Compiling model...'
	gc.collect()
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)

	print 'Compiling model...'
	model.compile(loss='mean_squared_error',
			  optimizer='sgd',
			  metrics=['accuracy'])

	print 'Loading dictionary...'

	with open('../dataset/temporal_test_data.pickle','rb') as f1:
		temporal_test_data=pickle.load(f1)
	evaluate(model, temporal_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size)

if __name__ == "__main__":
	CNN()