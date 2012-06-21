# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:23:24 2012

@author: mba
"""

# Exercise 1.1
import sys, os
start_path = os.getcwd()
os.chdir("C:\Users\mba\LxMLS_2012\lxmls-toolkit\lxmls")
sys.path.append("readers")
sys.path.append("classifiers")
sys.path.append("distributions")

import simple_data_set as sds
import linear_classifier as lcc
import gaussian_naive_bayes as gnbc
import naive_bayes as nb

sd = sds.SimpleDataSet(nr_examples=100, g1 = [[-1,-1],1], g2 = [[1,1],1], balance = 0.5, split=[0.5,0,0.5])

fig,axis = sd.plot_data()

gnb = gnbc.GaussianNaiveBayes()
params_nb_sd = gnb.train(sd.train_X, sd.train_y)

print "Estimated Means"
print gnb.means
print "Estimated Priors"
print gnb.prior
y_pred_train = gnb.test(sd.train_X,params_nb_sd)
acc_train = gnb.evaluate(sd.train_y, y_pred_train)
y_pred_test = gnb.test(sd.test_X,params_nb_sd)
acc_test = gnb.evaluate(sd.test_y, y_pred_test)
print "Gaussian Naive Bayes Simple Dataset Accuracy"
print "train: %f test: %f" % (acc_train,acc_test)

fig,axis = sd.add_line(fig,axis,params_nb_sd,"Naive Bayes","red")

# Exercise 1.2
import sentiment_reader as srs
import naive_bayes as nb

scr = srs.SentimentCorpus("books")

import MultinomialNaiveBayes as mnbb

mnb = mnbb.MultinomialNaiveBayes()
params_nb_sc = mnb.train(scr.train_X,scr.train_y)
y_pred_train = mnb.test(scr.train_X,params_nb_sc)
acc_train = mnb.evaluate(scr.train_y, y_pred_train)
y_pred_test = mnb.test(scr.test_X,params_nb_sc)
acc_test = mnb.evaluate(scr.test_y, y_pred_test)
print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test) 

os.chdir(start_path)
