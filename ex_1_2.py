# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:23:24 2012

@author: mba
"""

# From Exercise 1.1
import sys, os
start_path = os.getcwd()
os.chdir("/home/miguel/LxMLS/lxmls-toolkit/lxmls")
sys.path.append("readers")
sys.path.append("classifiers")
sys.path.append("distributions")

# Exercise 1.2
import sentiment_reader as srs
import naive_bayes as nb

scr = srs.SentimentCorpus("books")

os.chdir(start_path)
