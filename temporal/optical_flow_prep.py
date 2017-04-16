import cv2
import numpy as np
import pickle
from PIL import Image
import os
import gc
from skimage.io import imread
from skimage.io import imsave

def stackOpticalFlow(blocks,temporal_train_data,img_rows,img_cols, trainOrTesting):
	firstTime=1
	lessActivity = {'HandStandPushups':1,'JumpingJack':2,'PushUps':3}

	try:
		firstTimeOuter=1
		for block in blocks:
			fx = []
			fy = []
			filename,blockNo=block.split('@')
			path = 'of_images/' + trainOrTesting + '/'+filename
			blockNo=int(blockNo)
			
			if not filename.split('_')[1] in lessActivity:
				for i in range(0,10):
					imgH=imread(path+'/'+'h'+str(i*3+blockNo)+'_'+str(filename)+'.jpg')
					imgV=imread(path+'/'+'v'+str(i*3+blockNo)+'_'+str(filename)+'.jpg')
					imgH=cv2.resize(imgH, (img_rows,img_cols))
					imgV=cv2.resize(imgV, (img_rows,img_cols))
					fx.append(imgH)
					fy.append(imgV)

			elif filename.split('_')[1] in lessActivity:
				for i in range(0,10):
					imgH=imread(path+'/'+'h'+str(i+blockNo)+'_'+str(filename)+'.jpg')
					imgV=imread(path+'/'+'v'+str(i+blockNo)+'_'+str(filename)+'.jpg')
					imgH=cv2.resize(imgH, (img_rows,img_cols))
					imgV=cv2.resize(imgV, (img_rows,img_cols))
					fx.append(imgH)
					fy.append(imgV)

			flowX = np.dstack((fx[0],fx[1],fx[2],fx[3],fx[4],fx[5],fx[6],fx[7],fx[8],fx[9]))
			flowY = np.dstack((fy[0],fy[1],fy[2],fy[3],fy[4],fy[5],fy[6],fy[7],fy[8],fy[9]))
			inp = np.dstack((flowX,flowY))
			inp = np.expand_dims(inp, axis=0)
			if not firstTime:	
				inputVec = np.concatenate((inputVec,inp))
				labels=np.append(labels,temporal_train_data[block]-1)
			else:
				inputVec = inp
				labels=np.array(temporal_train_data[block]-1)
				firstTime = 0

		inputVec=np.rollaxis(inputVec,3,1)
		inputVec=inputVec.astype('float16',copy=False)
		labels=labels.astype('int',copy=False)
		gc.collect()

		return (inputVec,labels)
	except:
		return (None,None)


def writeOpticalFlow(path,filename,w,h,c):
	count=0
	try:
		cap = cv2.VideoCapture(path+'/'+filename)
		ret, frame1 = cap.read()

		if frame1==None:
			return count

		frame1 = cv2.resize(frame1, (w,h))
		prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

		folder = './of_images'+'/'+filename+'/'
		dir = os.path.dirname(folder)
		os.mkdir(dir)

		while(1):
			ret, frame2 = cap.read()

			if frame2==None:
				break
			count+=1
			if count%5==0:
				print (filename+':' +str(c)+'-'+str(count))

				frame2 = cv2.resize(frame2, (w,h))
				next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

				flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

				horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
				vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
				horz = horz.astype('uint8')
				vert = vert.astype('uint8')

				cv2.imwrite(folder+'h'+str(count)+'_'+filename+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
				cv2.imwrite(folder+'v'+str(count)+'_'+filename+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
				
				prvs = next

		cap.release()
		cv2.destroyAllWindows()
		return count
	except Exception,e:
		print e
		return count