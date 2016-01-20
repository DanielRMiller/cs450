# We will need this import to read the data from the URL
import urllib.request
# Needed to turn my string into a file format
import io
# Args
import sys
# Numpy arrays, etc
import numpy
# To load iris if no URL or file name
from sklearn import datasets
# We will need this to get the data from a csv file format to an array/list
import csv
# My personal Classifiers
from hCClassifier import HCClassifier
from kNNClassifier import KNNClassifier
# For comparison
from sklearn.neighbors import KNeighborsClassifier

def main(argv):
	# Load a dataset containing many instances each with a set of attributes and a target value.
	if len(argv) >= 2:
		fileOrURL = argv[1]
		isFile = False if fileOrURL.startswith('http') else True
		if(isFile):
			# Now we are loading data from a data file that is saved on our computer.
			with open(fileOrURL, 'r') as myfile:
				f=myfile.read()

		else:
			# Lets open the data from the url that has the data in a csv file format
			# fileOrURL = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
			f = urllib.request.urlopen(fileOrURL).read().decode('utf-8')[:-1]

		raw = list(csv.reader(f.split('\n'), delimiter=','))[:-1]
		data = []
		targets = []
		for sublist in raw:
			# data (all but last col) goes into our data
			# Change data from strings to floats
			data.append([float(i) for i in sublist[:-1]])
			# target (just the last col) goes into our target
			targets.append(sublist[-1])
		# Make it a numpy array just like the other two are
		data = numpy.array(data)
		targets = numpy.array(targets)
	else:
		# Please use the popular Iris dataset (natively in scikit-learn).
		iris = datasets.load_iris()
		data = iris.data
		targets = iris.target

	# Randomize the order of the instances in the dataset. Don't forget that you need to keep the targets matched up with the approprite instance.
	# This permutation will be used for both the data and the target so it will line up correctly.
	perm = numpy.random.permutation(len(data))

	data = data[perm]
	targets = targets[perm]
	# Split the data into two sets: a training set (70%) and a testing set (30%)
	# Index of where to split (we want this to be an integer)
	index = int(round(perm.size*.3))
	test = perm[:index]
	train = perm[index:]

	# Instantiate your new classifier
	# classifier = HCClassifier()
	classifier = KNNClassifier(3)
	# "Train" it with data
	classifier.train(data[train], targets[train])
	# Make "Predictions" on the test data
	predictions = classifier.predict(data[test])
	# Reset correct answers
	correct = 0
	# Count the answers that we got right
	for (prediction, actual) in zip(predictions, test):
		if prediction == targets[actual]:
			correct += 1
	#Determine the accuracy of your classifier's predictions (reported as percentage)
	print (correct/test.size*100)
	# Create new public repository at GitHub and publish code
	# Github repository: http://github.com/DanielRMiller/cs450
	return

# To make sure main is ran only when ran, not loaded as well
if __name__ == "__main__":
    main(sys.argv)
