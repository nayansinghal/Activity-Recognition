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

def get_activations(model, layer, X_batch):
	get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
	activations = get_activations(X_batch)
	return activations

def getTrainData(chunk,nb_classes,img_rows,img_cols):
	X_train,Y_train=tsd.stackOFTrain(chunk,img_rows,img_cols)
	if (X_train!=None and Y_train!=None):
		X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

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

def CNN():
	input_frames=10
	batch_size=64
	nb_classes = 14
	nb_epoch = 200
	img_rows, img_cols = 150,150
	img_channels = 2*input_frames
	chunk_size=128
	print 'Loading dictionary...'

	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	with open('../dataset/temporal_test_data.pickle','rb') as f1:
		temporal_test_data=pickle.load(f1)

	print 'Preparing architecture...'
	model = createModel(img_channels, img_rows, img_cols, nb_classes, sys.argv[1])

	print 'Compiling model...'
	gc.collect()
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	model.compile(loss='categorical_crossentropy',optimizer=sgd)

	for e in range(2,nb_epoch):
		print('-'*40)
		print('Epoch', e)
		print('-'*40)
		instance_count=0

		train_keys=temporal_train_data.keys()
		random.shuffle(train_keys)

		for chunk in chunks(train_keys,chunk_size):
			print instance_count
			instance_count+=chunk_size
			print("Preparing training data...")
			X_batch,Y_batch=getTrainData(chunk,nb_classes,img_rows,img_cols)
			if (X_batch!=None and Y_batch!=None):
				loss = model.fit(X_batch, Y_batch, verbose=1, batch_size=batch_size, epochs=1)	

		model.save_weights('model/temporal_stream_model'+str(10*e+instance_count)+'.h5',overwrite=True)

		print("Preparing testing data...")
		test_keys=temporal_test_data.keys()
		random.shuffle(test_keys)

		for chunk in chunks(test_keys,chunk_size):
			X_test,Y_test=getTestData(chunk,nb_classes,img_rows,img_cols)
			if (X_batch!=None and Y_batch!=None):
				loss = model.evaluate(X_test,Y_test,batch_size=batch_size,verbose=1)
			print('Validation Loss:', loss)

if __name__ == "__main__":
	CNN()
