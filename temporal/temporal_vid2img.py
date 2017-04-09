import numpy as np
import sys,os
import pickle
import scandir
import gc
import cv2
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.io import imsave

def writeOpticalFlow(path,filename,w,h,c):
	count=0
	frame_no =1
	try:
		fileN = filename + "_" + str(frame_no) + ".jpg"	
		frame1 = imread(path + '/' + fileN)

		if frame1==None:
			return count

		frame1 = cv2.resize(frame1, (w,h))
		prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

		folder = './of_images'+'/'+filename+'/'
		dir = os.path.dirname(folder)
		os.mkdir(dir)

		while(1):
			frame_no = frame_no + 1
			fileN = filename + "_" + str(frame_no) + ".jpg"	
			frame2 = imread(path + '/' + fileN)

			if frame2==None:
				break
			count+=1

			frame2 = cv2.resize(frame2, (w,h))
			next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

			flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

			horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
			vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
			horz = horz.astype('uint8')
			vert = vert.astype('uint8')

			imsave(folder+'h'+str(count)+'_'+filename+'.jpg',horz)
			imsave(folder+'h'+str(count)+'_'+filename+'.jpg',vert)
				
			prvs = next

		cap.release()
		cv2.destroyAllWindows()
		return count
	except Exception,e:
		print e
		return count

def writeOF():

	root = "../videos/"
	w=224
	h=224
	c=0
	data={}

	for path, subdirs, files in scandir.walk(root):
		if len(subdirs) == 0:
			filename = path.split('/')[-1]
			print filename
			count=writeOpticalFlow(path,filename,w,h,c)
			if count:
				data[filename]=count
			c+=1
			with open("done.txt", "a") as myfile:
				myfile.write(filename+'-'+str(c)+'\n')

	with open('../dataset/frame_count.pickle','wb') as f:
		pickle.dump(data,f)

if __name__ == "__main__":
	writeOF()
