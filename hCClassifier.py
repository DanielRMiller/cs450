# Create a class for a "HardCoded" classifier
class HCClassifier:
	"""A class that is hardcoded to give a specific answer no matter the data"""
	# Train: Accepts a training instances and targets. Does nothing with them.
	def train(self, instances, targets): return
	# Predict: Accepts the test instances and returns a hard-coded response.
	def predict(self, instances):
		predictions = [];
		for instance in instances:
			predictions.append(0)
		return predictions;
	def fit(self, data, targets):
	    self.train(data, targets)