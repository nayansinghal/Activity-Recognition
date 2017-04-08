import cv2
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def get_sample_data(chunk, img_row, img_col):
	X_train = []
	Y_train = []
	with open('./spatial_train_data_new.pickle','rb') as f1:
		spatial_train_data=pickle.load(f1)
	for imgname in chunk:
		idx = imgname.rfind('_')
		folder = imgname[:idx]
		filename = './sp_images/train'+'/'+folder+'/'+imgname+'.jpg'
		img = imread(filename)
		if img != None:
			img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
			X_train.append(img)
			Y_train.append(spatial_train_data[imgname])

	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)
	return X_train,Y_train


def get_train_data(chunk, img_row, img_col):
	X_train = []
	Y_train = []
	with open('spatial_train_data_new.pickle','rb') as f1:
		spatial_train_data=pickle.load(f1)
	try:
		for imgname in chunk:
			Y_train.append(spatial_train_data[imgname])
			idx = imgname.rfind('_')
			folder = imgname[:idx]
			filename = './sp_images/train'+'/'+folder+'/'+imgname+'.jpg'
			img = imread(filename)
			img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
			X_train.append(img)

		X_train = np.asarray(X_train)
		Y_train = np.asarray(Y_train)
		return X_train,Y_train
	except:
		X_train=None
		Y_train=None
		return X_train,Y_train

def get_test_data(chunk, img_row, img_col):
	X_train = []
	Y_train = []
	with open('spatial_test_data_new.pickle','rb') as f1:
		spatial_test_data=pickle.load(f1)
	try:
		for imgname in chunk:
			Y_train.append(spatial_test_data[imgname])
			idx = imgname.rfind('_')
			folder = imgname[:idx]
			filename = './sp_images/test'+'/'+folder+'/'+imgname+'.jpg'
			img = imread(filename)
			img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
			X_train.append(img)

		X_train = np.asarray(X_train)
		Y_train = np.asarray(Y_train)
		
		return X_train,Y_train
	except:
		X_train=None
		Y_train=None
		return X_train,Y_train

if __name__ == '__main__':
	get_data()          
