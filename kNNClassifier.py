import numpy
from collections import Counter


def euclidean_distance(instance1, instance2):
	"""Calculate squared distance"""
	# Reset the distance
	distance = 0
	for i in range(len(instance1)):
		# Positive numbers = find squared distance
		if instance1[i] >= 0:
			distance += (instance1[i] - instance2[i]) ** 2
		# when not the same and negative add one
		else:
			if instance1[i] != instance2[i]:
				distance += 1
	return distance

class KNNClassifier:
	"""An implementation of the k nearest neighbor algorithm"""
	def __init__(self, k=1):
		# Number of neighbors
		self.k = k
		# Variables to add to columns
		self.std_devs = self.means = []
		# Original Data
		self.data = self.targets = ""
	def train(self, data, targets):
		# Store the information for recall
		self.data = data
		self.targets = targets
		# Normalize all data
		for i in range(len(self.data[0])):
			# Add standard deviation column
			self.std_devs.append(numpy.std(self.data[:, i]))
			# Add the mean column
			self.means.append(numpy.mean(self.data[:, i]))
			# Reduce means 
			self.data[:, i] -= self.means[i]
			# Reduce std_devs
			self.data[:, i] /= self.std_devs[i]
	def predict(self, data):
		# This is what we are going to predict
		prediction = []
		# For each instance in the data set
		for instance in data:
			# Normalize the instance
			for i, val in enumerate(instance):
				instance[i] -= self.means[i]
				instance[i] /= self.std_devs[i]
			# Reset the distances
			distances = []
			# Create the distance array
			for myInstance in self.data:
				distances.append(euclidean_distance(myInstance, instance))
			# Find the nearest
			nearest = numpy.argsort(distances)
			# Reset the neighbors
			neighbors = []
			# Append the K nearest
			for i in range(min(self.k, len(nearest))):
				neighbors.append(self.targets[nearest[i]])
			# Append only the K nearest neighbors
			prediction.append(Counter(neighbors).most_common()[0][0])
		# Make like a tree
		return prediction
	def fit(self, data, targets):
		# Just to make it comparable to sklearn
		self.train(data, targets)
