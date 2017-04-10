import numpy as np
import cv2
import sys
import os
import pickle
import shutil
import gc
from skvideo.io import vread
from skvideo.io import vreader
from skvideo.io import vwrite

def write_images():

	root = 'sp_images_20/'

	classes = open('../dataset/classInd_14.txt','r')
	var1 = {}
	for line in classes:
		words = line.split(" ")
		var1[words[1].split("\n")[0]] = words[0]

	for path, subdirs, files in os.walk(root):
		for filename in files:
			print filename
			if ".DS_Store" not in filename:
				folder = 'im_train' + '/' + filename.split('.')[0] + '/'   
				if not os.path.isdir(folder):
					os.mkdir(folder)
				else:
					shutil.rmtree(folder)
					os.mkdir(folder)
				try:
					cnt = 0
					full_path = path + '/' + filename
					cap = vreader(full_path)
					fcnt = 1

					for frame in cap:
						vid_name = filename.split('.')[0]
						img_path = folder + vid_name + '_{}.jpg'.format(cnt + 1)
						img_name = vid_name + '_{}'.format(cnt + 1)
						print 'image name:', img_name
						vwrite(img_path, frame)
						cnt = cnt + 1
						fcnt += 1

					if cnt:
						with open("count.txt", "a") as txt:
							text = str(cnt) + " " + img_name.split('.')[0] + "\n"
							txt.write(text)
				except (RuntimeError, TypeError, NameError):
					print "Some Error happened"

def data_prep():

	root = 'im_train/'
	path = os.path.join(root, "")

	classes = open('../dataset/classInd_14.txt','r')
	var1 = {}
	for line in classes:
		words = line.split(" ")
		var1[words[1].split("\n")[0]] = words[0]

	for path, subdirs, files in os.walk(root):
		for filename in files:
			if ".DS_Store" not in filename:
				frame_name = filename.split('.')[0]
				idx = frame_name.rfind('_')
				if (idx != -1) and (frame_name[:idx] in var1):
					vidname = frame_name[:idx]
					dic[frame_name] = var1[vidname]
					print vidno, 
					vidno+=1

	with open('./spatial_train_data_new.pickle', 'w') as f:
		pickle.dump(dic, f)

if __name__ == "__main__":
	write_images()
	gc.collect()
	#data_prep()