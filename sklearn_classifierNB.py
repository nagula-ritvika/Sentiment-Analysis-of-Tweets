import numpy
from sklearn.naive_bayes import GaussianNB

__author__ = 'Ritvika'
from FeatureWords import *

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
    count =0
    for row in inpTweets:
        sentiment = row[0]
        tweet = row[1]
        featureVector = getFeatureVector(processTweet(tweet))
        train_data.append(' '.join(featureVector))
        train_labels.append(sentiment)
        count+=1
    print "trainData",train_data

def extractTestData():
    inpTestTweets =  csv.reader(open("FlipkartTestTweets.csv",'rb'),delimiter=',')
    for row in inpTestTweets:
        sentiment = row[0]
        tweet = row[1]
        featureVector = getFeatureVector(processTweet(tweet))
        test_data.append(' '.join(featureVector))
        test_labels.append(sentiment)
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

X_train = numpy.array(train_data)
Y_train = numpy.array(train_labels)
max1 = 0
for i in range(0,len(X_train)):
    l=len(X_train[i].split(" "))
    if max1<l:
        max1=l
print max1
print (X_train).shape
print (Y_train).shape
print Y_train
X1_train=X_train.reshape(len(X_train),6)
print  X1_train.shape

X_dev_test = numpy.array(dev_test_data)
Y_dev_test = numpy.array(dev_test_labels)

X_test = numpy.array(test_data)
Y_test = numpy.array(test_labels)

classifier_NB= GaussianNB()
print "classifier created"
classifier_NB.fit(X_train,Y_train)
print "classifier fitted"
print classifier_NB.predict(test_data)
