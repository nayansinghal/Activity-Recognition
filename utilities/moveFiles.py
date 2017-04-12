import os
import sys
import shutil

'''
  argv[1] : Training or testing file
  argv[2] : Source Directory Path
  argv[3] : Destination Directory Path
'''
if __name__ == "__main__":

	input = open(sys.argv[1], "r")
	srcPath = sys.argv[2]
	desPath = sys.argv[3]
	print srcPath
	print desPath

	for line in input:
		srcFile = srcPath  + line.split(" ")[0].split("/")[1].split(".avi")[0]
		desFile = desPath  + line.split(" ")[0].split("/")[1].split(".avi")[0]

		if os.path.isdir(srcFile):
			if not os.path.isdir(desFile):
				os.mkdir(desFile)
				shutil.move(srcFile, desFile)

	input.close()