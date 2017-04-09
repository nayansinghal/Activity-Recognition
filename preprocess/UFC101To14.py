import os
import sys

if __name__ == "__main__":

	classes = open('classInd.txt','r')
	dic = {}
	for line in classes:
		words = line.split(" ")
		dic[words[1].split("\n")[0]] = words[0]

	filename = sys.argv[1]

	input = open(filename, "r")
	output = open("output/" + filename, "w")

	for line in input:
		words = line.split(" ")
		actName = line.split("/")[0]
		if actName in dic:
			output.write(words[0].split("\n")[0].split("\r")[0] + " " + dic[actName] + "\n")

	input.close()
	output.close()


	
