import numpy as np
import scipy
import scipy.stats

class kNN():

	def __init__(self, k=1, distance_m=None):
		self._k = k
		self.distance = distance_m

	def set_distance_measure(self,distance_m):
		self.distance = distance_m

	def _majority_label(self,Y):
		return scipy.stats.mode(Y).mode[0]

	def fit(self, X, Y):

		if (X.shape[0] == 0 ):
			raise ValueError ('No samples provided')

		self._n_features = len(X[0])
		if all([len(ii) == self._n_features for ii in X]) == False:
			raise ValueError ('Sample size is not same every sample')

		if len(X) != len(Y):
			raise ValueError ('Length Mismatch between X and Y')


		self._X = X
		self._Y = Y

	def predict(self,X):

		try:
			if len(X.shape) == 1:
				X = X.reshape(1,-1)

			if (self._X.shape[1] != X.shape[1]):
				raise ValueError('Shape mismatch for test sample ')

			if self.distance == None:
				raise ValueError('Set distance measure first')

			distM = []
			for item in X:
				curY = []
				for row in self._X:
					curY.append(self.distance(row,item))
				distM.append(curY)

			pred = []

			for item in distM:
				voters_index = np.argpartition(item,self._k)[0:self._k]
				votes = self._Y[voters_index]
				pred.append(self._majority_label(votes))

			return pred
		except AttributeError:
			raise AttributeError('Call fit method first')


