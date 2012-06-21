# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:23:24 2012

@author: mba
"""

############ Exercise 1.1
import sys, os
import matplotlib.pyplot as plt
start_path = os.getcwd()
os.chdir("C:\\Users\\mba\\LxMLS_2012\\lxmls-toolkit\\lxmls")
sys.path.append("readers")
sys.path.append("classifiers")
sys.path.append("distributions")
sys.path.append("util")

import simple_data_set as sds
import linear_classifier as lcc
import gaussian_naive_bayes as gnbc
import naive_bayes as nb

sd = sds.SimpleDataSet(nr_examples=100, g1 = [[-1,-1],1], g2 = [[1,1],1], balance = 0.5, split=[0.5,0,0.5])

fig,axis = sd.plot_data()
#plt.show()

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
print "********************************************"

fig,axis = sd.add_line(fig,axis,params_nb_sd,"Naive Bayes","red")
#plt.show()

############ Exercise 1.2
import sentiment_reader as srs
import naive_bayes as nb

scr = srs.SentimentCorpus("books")

import multinomial_naive_bayes as mnbc

mnb = mnbc.MultinomialNaiveBayes()
params_nb_sc = mnb.train(scr.train_X,scr.train_y)
y_pred_train = mnb.test(scr.train_X,params_nb_sc)
acc_train = mnb.evaluate(scr.train_y, y_pred_train)
y_pred_test = mnb.test(scr.test_X,params_nb_sc)
acc_test = mnb.evaluate(scr.test_y, y_pred_test)
print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test) 
print "********************************************"

############ Exercise 1.4
import perceptron as percc

perc = percc.Perceptron()
params_perc_sd = perc.train(sd.train_X,sd.train_y)
y_pred_train = perc.test(sd.train_X,params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)
y_pred_test = perc.test(sd.test_X,params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)
print "Perceptron Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)

fig,axis = sd.add_line(fig,axis,params_perc_sd,"Perceptron","blue")

print "********************************************"

############ Exercise 1.4, now for Amazon corpus
perc = percc.Perceptron()
params_perc_sd = perc.train(scr.train_X,sd.train_y)
y_pred_train = perc.test(scr.train_X,params_perc_sd)
acc_train = perc.evaluate(scr.train_y, y_pred_train)
y_pred_test = perc.test(scr.test_X,params_perc_sd)
acc_test = perc.evaluate(scr.test_y, y_pred_test)
print "Perceptron Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

print "********************************************"

############ Exercise 1.5: MIRA
import mira as mirac

mira = mirac.Mira()
mira.regularizer = 1.0 # This is lambda
params_mira_sd = mira.train(sd.train_X,sd.train_y)
y_pred_train = mira.test(sd.train_X,params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)
y_pred_test = mira.test(sd.test_X,params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)
print "Mira Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)

fig,axis = sd.add_line(fig,axis,params_mira_sd,"Mira","green")

print "********************************************"

############ Exercise 1.5: MIRA, now for Amazon corpus
params_mira_sc = mira.train(scr.train_X,scr.train_y)
y_pred_train = mira.test(scr.train_X,params_mira_sc)
acc_train = mira.evaluate(scr.train_y, y_pred_train)
y_pred_test = mira.test(scr.test_X,params_mira_sc)
acc_test = mira.evaluate(scr.test_y, y_pred_test)

print "Mira Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

print "********************************************"

############ Exercise 1.6: MaxEnt with L-BFGS
import max_ent_batch as mebc

me_lbfgs = mebc.MaxEnt_batch()
me_lbfgs.regularizer = 1.0
params_meb_sd = me_lbfgs.train(sd.train_X,sd.train_y)
y_pred_train = me_lbfgs.test(sd.train_X,params_meb_sd)
acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(sd.test_X,params_meb_sd)
acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)
print "Max-Ent Batch Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)

fig,axis = sd.add_line(fig,axis,params_meb_sd,"Max-Ent-Batch","orange")

print "********************************************"

############ Exercise 1.6: MaxEnt with L-BFGS, now for Amazon corpus
params_meb_sc = me_lbfgs.train(scr.train_X,scr.train_y)
y_pred_train = me_lbfgs.test(scr.train_X,params_meb_sc)
acc_train = me_lbfgs.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(scr.test_X,params_meb_sc)
acc_test = me_lbfgs.evaluate(scr.test_y, y_pred_test)
print "Max-Ent Batch Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

print "********************************************"

############ Exercise 1.6: MaxEnt with stochastic gradient
import max_ent_online as meoc

me_sgd = meoc.MaxEnt_online()
me_sgd.regularizer = 1.0
params_meo_sc = me_sgd.train(scr.train_X,scr.train_y)
y_pred_train = me_sgd.test(scr.train_X,params_meo_sc)
acc_train = me_sgd.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_sgd.test(scr.test_X,params_meo_sc)
acc_test = me_sgd.evaluate(scr.test_y, y_pred_test)
print "Max-Ent Online Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

print "********************************************"

############ Exercise 1.7: SVM
import svm as svmc

svm = svmc.SVM()
svm.regularizer = 1.0 # This is lambda
params_svm_sd = svm.train(sd.train_X,sd.train_y)
y_pred_train = svm.test(sd.train_X,params_svm_sd)
acc_train = svm.evaluate(sd.train_y, y_pred_train)
y_pred_test = svm.test(sd.test_X,params_svm_sd)
acc_test = svm.evaluate(sd.test_y, y_pred_test)
print "SVM Online Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)

fig,axis = sd.add_line(fig,axis,params_svm_sd,"SVM","magenta")

print "********************************************"

############ Exercise 1.7: SVM, now for Amazon corpus
params_svm_sc = svm.train(scr.train_X,scr.train_y)
y_pred_train = svm.test(scr.train_X,params_svm_sc)
acc_train = svm.evaluate(scr.train_y, y_pred_train)
y_pred_test = svm.test(scr.test_X,params_svm_sc)
acc_test = svm.evaluate(scr.test_y, y_pred_test)
print "SVM Online Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

print "********************************************"

plt.show() # Flushes the figure buffer
os.chdir(start_path)
