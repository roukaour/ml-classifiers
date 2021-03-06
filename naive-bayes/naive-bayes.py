#!/usr/bin/python

from __future__ import print_function, division

from collections import namedtuple, defaultdict
from math import log
from itertools import product

__author__ = 'Remy Oukaour (107122849)'

Instance = namedtuple('Instance', ['label', 'features'])

Classification = namedtuple('Classification', ['predicted', 'true'])

class NaiveBayesClassifier(object):

	# Entropy threshold for feature selection
	min_entropy = 0.15

	# Parameter used for Laplace smoothing of feature value likelihoods
	smoothing_value = 0.001

	def __init__(self):
		self.num_instances = 0
		self.label_num = defaultdict(lambda: 0)
		self.fvl_num = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
		self.feature_values = defaultdict(lambda: set())
		self.feature_entropy = defaultdict(lambda: 0)

	def labels(self):
		return list(sorted(self.label_num.keys()))

	def train(self, instances):
		for instance in instances:
			self.num_instances += 1
			self.label_num[instance.label] += 1
			for f, v in enumerate(instance.features):
				self.fvl_num[f][v][instance.label] += 1
				self.feature_values[f].add(v)

	def classify(self, instances):
		for instance in instances:
			yield Classification(self.classify_features(instance.features), instance.label)

	def classify_features(self, features):
		# Maximum a posteriori estimation: arg max(for each label: P(features|label))
		return max(self.labels(), key=lambda label: self.log_posterior(features, label))

	def log_posterior(self, features, label):
		# P(label|features) = P(label) * P(features|label) / P(features)
		# log P(label|features) = log P(label) + log P(features|label) - log P(features)
		# (we leave out the evidence value, P(features), since it is constant)
		return self.log_prior(label) + self.log_likelihood(features, label)

	def log_prior(self, label):
		# P(label) = # instances with label / # instances
		# log P(label) = log # instances with label - log # instances
		return log(self.label_num[label]) - log(self.num_instances)

	def log_likelihood(self, features, label):
		# P(features|label) = product(for each feature: P(feature value|label))
		# log P(features|label) = sum(for each feature: log P(feature value|label))
		return sum(self.log_feature_likelihood(f, v, label) for f, v in enumerate(features)
			if self.entropy(f) > self.min_entropy)

	def log_feature_likelihood(self, feature, value, label):
		# P(feature value|label) = # instances with label where feature has value / # instances with label
		# log P(feature value|label) = log # instances with label where feature has value - log # instances with label
		# (we use Laplace smoothing to handle zero counts, necessary since log(0) is undefined)
		return (log(self.fvl_num[feature][value][label] + self.smoothing_value) -
			log(self.label_num[label] + self.smoothing_value * len(self.feature_values[feature])))

	def entropy(self, feature):
		if feature in self.feature_entropy:
			return self.feature_entropy[feature]
		# H(feature) = -sum(for each feature value: P(value) * log P(value))
		# P(value) = # feature with value / # feature
		# log P(value) = log # feature with value - log # feature
		H = 0
		for value in self.fvl_num[feature]:
			fv_total = sum(self.fvl_num[feature][value].values())
			if fv_total > 0:
				H -= (fv_total / self.num_instances) * (log(fv_total) - log(self.num_instances))
		self.feature_entropy[feature] = H
		return H

# The width and height of an image
IMAGE_SIZE = 28

# The background pixel character for images
IMAGE_BG = ' '

def num_regions(image):
	# Count the contiguous foreground/background regions in an image
	def flood_fill(filled, i, j, bg):
		if (0 <= i < IMAGE_SIZE and 0 <= j < IMAGE_SIZE and not filled[i][j] and
			(image[i][j] == IMAGE_BG) == bg):
			filled[i][j] = True
			for di, dj in product([-1, 0, 1], [-1, 0, 1]):
				flood_fill(filled, i+di, j+dj, bg)
	filled = [[False] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
	n = 0
	for i, j in product(range(IMAGE_SIZE), range(IMAGE_SIZE)):
		if not filled[i][j]:
			n += 1
			flood_fill(filled, i, j, image[i][j] == IMAGE_BG)
	return n

def spread_ratio(image):
	# Return the ratio of vertical to horizontal foreground spread
	hds, vds = [], []
	for i, j in product(range(IMAGE_SIZE), range(IMAGE_SIZE)):
		if image[i][j] != IMAGE_BG:
			hds.append(abs(i - IMAGE_SIZE / 2))
			vds.append(abs(j - IMAGE_SIZE / 2))
	return sum(vds) // sum(hds)

def vertical_bias(image):
	# Return the difference between the top and bottom background areas in an image
	top, bottom = sum(image[:IMAGE_SIZE//2], ()), sum(image[IMAGE_SIZE//2:], ())
	return top.count(IMAGE_BG) - bottom.count(IMAGE_BG)

def horizontal_bias(image):
	# Return the difference between the left and right background areas in an image
	mirror = zip(*image)
	left, right = sum(mirror[:IMAGE_SIZE//2], ()), sum(mirror[IMAGE_SIZE//2:], ())
	return left.count(IMAGE_BG) - right.count(IMAGE_BG)

def digit_instances(label_filename, feature_filename):
	with open(label_filename, 'r') as label_file, open(feature_filename, 'r') as feature_file:
		for label_line in label_file:
			label = int(label_line)
			image = [tuple(feature_file.readline().rstrip('\n')) for _ in range(IMAGE_SIZE)]
			# Use some holistic quantities as features
			features = [num_regions(image), spread_ratio(image),
				(vertical_bias(image), horizontal_bias(image))]
			# Use overlapping 2x2 pixel blocks as features
			for i, j in product(range(IMAGE_SIZE-1), range(IMAGE_SIZE-1)):
				feature = image[i][j:j+2] + image[i+1][j:j+2]
				features.append(feature)
			yield Instance(label, features)

def build_confusion_matrix(data, labels):
	n = len(labels)
	# Initialize confusion matrix
	cm = [[''] * (n + 3) for _ in range(n + 3)]
	# Add row and column headers
	cm[0][0] = 'pred\\true'
	cm[0][-2] = cm[-2][0] = 'total'
	cm[0][-1] = 'precision'
	cm[-1][0] = 'sensitivity'
	cm[-2][-1] = 'accuracy'
	# Add label headers
	for i, label in enumerate(labels):
		cm[0][i+1] = cm[i+1][0] = label
	# Add data
	for (i, li), (j, lj) in product(enumerate(labels), enumerate(labels)):
		cm[i+1][j+1] = data[li][lj]
	# Calculate totals, precision, and sensitivity scores
	for i in range(n):
		cm[i+1][-2] = sum(cm[i+1][1:-2]) # row total
		cm[i+1][-1] = cm[i+1][i+1] / cm[i+1][-2] # precision
		cm[-2][i+1] = sum(cm[j+1][i+1] for j in range(n)) # column total
		cm[-1][i+1] = cm[i+1][i+1] / cm[-2][i+1] # sensitivity
	# Calculate accuracy
	cm[-2][-2] = sum(cm[-2][1:-2]) # total instances
	cm[-1][-2] = sum(cm[i+1][i+1] for i in range(n)) # correct predictions
	cm[-1][-1] = cm[-1][-2] / cm[-2][-2] # accuracy
	return cm

def main():
	# Train naive Bayes classifier on training data
	nbc = NaiveBayesClassifier()
	training_data = digit_instances('traininglabels.txt', 'trainingimages.txt')
	nbc.train(training_data)
	# Classify test data and output classifications to a file
	test_data = digit_instances('testlabels.txt', 'testimages.txt')
	classifications = nbc.classify(test_data)
	results = defaultdict(lambda: defaultdict(lambda: 0))
	with open('predictedlabels.txt', 'w') as prediction_file:
		for result in classifications:
			results[result.predicted][result.true] += 1
			prediction_file.write('%d\n' % result.predicted)
	# Build a confusion matrix from the classifications and output it
	confusion_matrix = build_confusion_matrix(results, nbc.labels())
	for row in confusion_matrix:
		print(*row, sep='\t')

if __name__ == '__main__':
	main()
