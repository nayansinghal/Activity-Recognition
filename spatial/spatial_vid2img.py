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

	root = '../UCF-14/'

	classes = open('../dataset/classInd_14.txt','r')
	var1 = {}
	for line in classes:
		words = line.split(" ")
		var1[words[1].split("\n")[0]] = words[0]

	for path, subdirs, files in os.walk(root):
		for filename in files:
			print filename
			if ".DS_Store" not in filename:
				folder = 'images' + '/' + filename.split('.')[0] + '/'
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
						if fcnt % 5 == 0:
							vwrite(img_path, frame)
							cnt = cnt + 1
						fcnt += 1

					if cnt:
						with open("count.txt", "w") as txt:
							text = str(cnt) + " " + img_name.split('.')[0] + "\n"
							txt.write(text)
				except (RuntimeError, TypeError, NameError):
					print "Some Error happened"

def data_prep():

	root = 'sp_images/test'
	path = os.path.join(root, "")

	classes = open('../dataset/classInd_14.txt','r')
	var1 = {}
	for line in classes:
		words = line.split(" ")
		var1[words[1].split("\n")[0]] = words[0]

	dic = {}
	vidno = 0

	for path, subdirs, files in os.walk(root):
		for filename in files:
			if ".DS_Store" not in filename:
				frame_name = filename.split('.')[0]
				idx = frame_name.rfind('_')
				#print frame_name, idx, frame_name[:idx].split("_")[1]
				if (idx != -1) and (frame_name[:idx].split("_")[1] in var1):
					print frame_name
					vidname = frame_name[:idx].split("_")[1]
					dic[frame_name] = var1[vidname]
					print vidno, 
					vidno+=1

	with open('./spatial_test_data_new.pickle', 'w') as f:
		pickle.dump(dic, f)

if __name__ == "__main__":
	write_images()
	gc.collect()
	data_prep()