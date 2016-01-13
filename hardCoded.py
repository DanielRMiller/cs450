# Load a dataset containing many instances each with a set of attributes and a target value.
# Please use the popular Iris dataset (natively in scikit-learn).
from sklearn import datasets
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that corespond to each number
#print(iris.target_names)

# Randomize the order of the instances in the dataset. Don't forget that you need to keep the targets matched up with the approprite instance.
import numpy as np
# This permutation will be used for both the data and the target so it will line up correctly.
perm = np.random.permutation(iris.target.size)
#print (iris.data[perm])
#print (iris.target[perm])

# Split the data into two sets: a training set (70%) and a testing set (30%)
# Index of where to split (we want this to be an integer)
index = int(round(perm.size*3/10))
#print ('index')
#print (index)
test = perm[:index]
#print (iris.target[test])
train = perm[index:]
#print (iris.target[train])
#print (perm.size)

# Create a class for a "HardCoded" classifier
class HardCoded:
	"""A class that is hardcoded to give a specific answer no matter the data"""
	# Train: Accepts a training instances and targets. Does nothing with them.
	def train(self, instances, targets): return
	# Predict: Accepts the test instances and returns a hard-coded response.
	def predict(self, instances):
		predictions = [];
		for instance in instances:
			predictions.append(0)
		return predictions;

# Instantiate your new classifier
hc = HardCoded()
# "Train" it with data
hc.train(iris.data[train], iris.target[train])
# Make "Predictions" on the test data
predictions = hc.predict(iris.data[test])
#print (predictions)
correct = 0
for (prediction, actual) in zip(predictions, train):
	#print ("prediction is: " + str(prediction))
	#print (iris.target[actual])
	#print (prediction == iris.target[actual])
	if prediction == iris.target[actual]:
		correct += 1
#print ('we got ' + str(correct) + " correct!")
#Determine the accuracy of your classifier's predictions (reported as percentage)
print (correct/test.size*100)
# Create new public repository at GitHub and publish code
# Github repository: http://github.com/DanielRMiller/cs450
