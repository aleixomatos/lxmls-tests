import numpy as np
import scipy as scipy
import linear_classifier as lc
from gaussian import *
#from distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.params = 0
        #self.variances = 0
        self.prior = 0
        
    def train(self,X,labels):
		classes = np.unique(labels) # list of all classes in the data
		Nclasses = classes.shape[0] # number of classes
		params = np.zeros(Nclasses) # initialization of the multinomial's parameters
		prior = np.zeros(Nclasses) # initialization of the class priors
		for i in range(Nclasses):
			idx = np.nonzero(labels == classes[i]) # check which points belong to this class
			prior[i] = 1.0*len(idx)/len(labels)
			idx = idx[0] # idx now has the index of points of class i
			params[i] = np.mean(X[idx]) # X[idx] contains the data with those indices
		self.prior = prior
		self.trained = True
		self.params = params
		return params
