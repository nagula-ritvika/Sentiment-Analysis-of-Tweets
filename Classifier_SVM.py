import csv
import numpy as np
from FeatureWords import *

__author__ = 'Ritvika'
import os
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

classes = ['positive', 'negative','neutral']
global train_data
global train_labels
global dev_test_data
global dev_test_labels
global test_data
global test_labels
# Read the data
train_data = []
train_labels = []
dev_test_data = []
dev_test_labels = []
test_data = []
test_labels = []


def extractTrainData():
    inpTweets = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    for row in inpTweets:
        sentiment = row[0]
        tweet = row[1]
        featureVector = getFeatureVector(processTweet(tweet))
        train_data.append(' '.join(featureVector))
        train_labels.append(sentiment)
    print type(train_data)
    print "trainData",train_data

def extractTestData():
    inpTestTweets =  csv.reader(open("FlipkartTestData.csv",'rb'),delimiter=',')
    for row in inpTestTweets:
        tweet = row[0]
        featureVector = getFeatureVector(processTweet(tweet))
        test_data.append(' '.join(featureVector))

    print "testData = ", test_data

def extractDevTestData():
    inpTestTweets =  csv.reader(open("flipkartDevTestData.csv",'rb'),delimiter=',')
    for row in inpTestTweets:
        sentiment = row[0]
        tweet = row[1]
        featureVector = getFeatureVector(processTweet(tweet))
        dev_test_data.append(' '.join(featureVector))
        dev_test_labels.append(sentiment)
    print "devTestData = ", dev_test_data

extractTrainData()
extractDevTestData()
extractTestData()


    # Create feature vectors
vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True, decode_error='ignore')
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)
dev_test_vectors = vectorizer.transform(dev_test_data)
# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
err_pred_rbf = classifier_rbf.predict(dev_test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
err_pred_linear = classifier_linear.predict(dev_test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
err_pred_liblinear = classifier_liblinear.predict(dev_test_vectors)

t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

def classifySVM(test_vectors, classifier):
    #op = open("SVM_Predictions.txt",'w')
    inp = csv.reader(open("FlipkartTestData.csv",'rb'),delimiter=',')
    i = 0
    for line in inp:
        testTweet = line[0]
        sentiment = classifier.predict(test_vectors[i])
        test_labels.append(np.array_str(sentiment).toString)
        #op.write(str(sentiment)+"\t"+testTweet+"\n")
        #print "test Tweet =",testTweet,"SVM classifier = ",sentiment
        i+=1
    print test_labels
    #op.close()

classifySVM(test_vectors,classifier_liblinear)
# Print results in a nice table
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_labels, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))


