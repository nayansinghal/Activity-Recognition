import numpy as np
import sys,os
import pickle
import optical_flow_prep as ofp
import gc

def stackOFTrain(chunk,img_rows,img_cols):
	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	X_train,Y_train=ofp.stackOpticalFlow(chunk,temporal_train_data,img_rows,img_cols, 'train')
	gc.collect()
	return (X_train,Y_train)

def stackOFTest(chunk,img_rows,img_cols):
	with open('../dataset/temporal_test_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	X_train,Y_train=ofp.stackOpticalFlow(chunk,temporal_train_data,img_rows,img_cols, 'test')
	gc.collect()
	return (X_train,Y_train)
