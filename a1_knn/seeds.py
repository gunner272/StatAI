import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.cross_validation import StratifiedKFold
import knn


def distance(X,Y):

	diff = X-Y
	return np.sqrt(np.dot(diff,diff))

def results(xtrain,xtest,ytrain,ytest,k):
	print 'Results for Knn with k =',k
	
	clf = knn.kNN(k=k,distance_m = distance)
	clf.fit(xtrain.values,ytrain.values)
	prd = clf.predict(xtest.values)
	
	print "Accuracy:",accuracy_score(ytest.values,prd)
	print 'Confusion Matrix'
	print confusion_matrix(ytest.values,prd)
	return accuracy_score(ytest.values,prd)

def result(xtrain,xtest,ytrain,ytest,k):
	print 'Results for Knn with k =',k
	
	clf = knn.kNN(k=k,distance_m = distance)
	clf.fit(xtrain,ytrain)
	prd = clf.predict(xtest)
	
	print "Accuracy:",accuracy_score(ytest,prd)
	print 'Confusion Matrix'
	print confusion_matrix(ytest,prd)
	return accuracy_score(ytest,prd)
	


def main():

	#df = pd.read_csv('../data/seeds.data',error_bad_lines = False,sep = '\t')
	#df.columns=['area','perimeter','compactness','k_length','k_width','assy_coef','g_length','label']

	df = pd.read_csv('../data/alabone.data',header = 0,error_bad_lines = False)

	tar = df['label']

	df = df.drop(['c1','label'],axis=1)
	# Q1 split 50-50%
	rk = {}
	rk[1] = []
	rk[2] = []
	rk[3] = []
	for i in range(0,10):
		print 'Test run',i
		xtrain,xtest,ytrain,ytest = tts(df,tar,test_size = 0.5)
 		rk[1].append(results(xtrain,xtest,ytrain,ytest,k=1))
 		print
 		rk[2].append(results(xtrain,xtest,ytrain,ytest,k=2))
 		print
		rk[3].append(results(xtrain,xtest,ytrain,ytest,k=3))
 			   
 	print "Mean accuracy and variance over 10 runs with k = 1",np.mean(rk[1]),np.var(rk[1])
 	print
 	print "Mean accuracy and variance over 10 runs with k = 2",np.mean(rk[2]),np.var(rk[2])
	print
	print "Mean accuracy and variance over 10 runs with k = 3",np.mean(rk[3]),np.var(rk[3])

	'''
	Cross validation 5 fold
	'''

	sf = StratifiedKFold(tar,n_folds = 5)
	i = 1
	rk[3] = []
	for train,test in sf:
		print 'Fold',i
		i = i +1
		xtrain,xtest,ytrain,ytest = df.values[train],df.values[test],tar.values[train],tar.values[test]
 		print
 		rk[3].append(result(xtrain,xtest,ytrain,ytest,k=3))
 		
 	print	
 	print "Mean accuracy and variance over 5-folds",np.mean(rk[3]),np.var(rk[3])


main()



