import numpy as np


class kNN():

	def __init__(self, k=1, distance_m=None):
		self._k = k
		self.distance = distance_m

	def set_distance_measure(self,distance_m):
		self.distance = distance_m

	def fit(self, X, Y):

		if (X.size[0] == 0 ):
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
			if (self._X.shape[1] != X.shape[1]):
				raise ValueError('Shape mismatch for test sample ')

			if self.distance == None:
				raise ValueError('Set distance measure first')

			distM = []
			for item in X:
				curY = []
				for index,row in enumerate(self._X):
					curY.append(distance(row,item))
				distM.append(curY)

			pred = []

			for item in distM:
				voters_index = np.argpartition(item,-1*self._k)[-1*self._k:]
				votes = Y[voters_index]
				pred.append(self._majority_label(votes))

			return pred
		except AttributeError:
			raise AttributeError('Call fit method first')


