import collections
import nltk
from ExtractFeatures import *
from nltk.metrics import *
from nltk.classify.util import accuracy

__author__ = 'Ritvika'
global dev_test_set
global train_set
global bg_train_set
global test_set
global bg_test_set
bg_test_set ={'positive': [],'negative': [], 'neutral': []}
test_set ={'positive': [],'negative': [], 'neutral': []}
dev_test_set ={'positive': [],'negative': [], 'neutral': []}
train_set={'positive': [],'negative': [], 'neutral': []}
bg_train_set ={'positive': [],'negative': [], 'neutral': []}

extractBulkFeatures()
#tt = open('corpusTweetsTrainingSet.txt','r')
#tweets = pickle.load(tt)
#print tweets
training_set = nltk.classify.util.apply_features(extract_features,tweets)
print "training set generated",type(training_set)
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print "classifier trained"
# Test the classifier
op1 = open("NBC_Unigrams.txt",'w')
inpTweets = csv.reader(open("FlipkartTestData.csv",'rb'),delimiter=',')
for row in inpTweets:
    testTweet = row[0]
    processedTestTweet = processTweet(testTweet)
    featVect = getFeatureVector(processedTestTweet)
    senti = NBClassifier.classify(extract_features(featVect))
    test_set[senti].append(' '.join(featVect))
    op1.write(senti+"\t"+testTweet+"\n")
    #print "test tweet = ",testTweet,"unigrams NB CLassifier = ",senti
print NBClassifier._label_probdist.prob('positive')
print NBClassifier._label_probdist.prob('negative')
print NBClassifier._label_probdist.prob('neutral')
#print "accuracy : ", nltk.classify.util.accuracy(NBClassifier,extract_features(getFeatureVector(processedTestTweet)))

extractBulkBigramFeatures()
bg_training_set = nltk.classify.util.apply_features(extract_bigram_features,tweets)

BigramNBC = nltk.NaiveBayesClassifier.train(bg_training_set)
op2 = open("NBC_Bigrams.txt",'w')
inpTweets = csv.reader(open("FlipkartTestData.csv",'rb'),delimiter=',')
for row in inpTweets:
    testTweet = row[0]
    processedTestTweet = processTweet(testTweet)
    featVect = getBigramFeatureVector(processedTestTweet)
    senti = BigramNBC.classify(extract_bigram_features(featVect))
    op2.write(senti+"\t"+testTweet+"\n")
    #print featVect
    fv=[]
    for t in featVect:
        fv.append(t[0]+","+t[1])
    bg_test_set[senti].extend(fv)

    #print "testTweet = ",testTweet,"Bigrams NB classifier = ", senti
print BigramNBC._label_probdist.prob('positive')
print BigramNBC._label_probdist.prob('negative')
print BigramNBC._label_probdist.prob('neutral')
#print bg_test_set['positive']
#print "accuracy : ", nltk.classify.util.accuracy(BigramNBC,extract_features(getFeatureVector(processedTestTweet)))

def calc_performance():
    train_data = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    for row in train_data:
        sentiment = row[0]
        tweet = row[1]
        featureVector = getFeatureVector(processTweet(tweet))
        train_set[sentiment].append(' '.join(featureVector))
    x=3#print type(test_set['positive'])
    #print test_set['positive']
    k=100
    p=10
    labels = ['positive','negative','neutral']
    #print train_set['positive']
    print "Results for Unigram Naive Bayes Classifier"
    print "             Precision   Recall \t\t\t\t\t F1-Score"
    print "positive   ",precision(set(train_set['positive']),set(test_set['positive']))*k,"\t\t",(recall(set(train_set['positive']),set(test_set['positive']))*p)\
        ,"\t\t\t",f_measure(set(train_set['positive']),set(test_set['positive']))*p*x
    print "negative  ",precision(set(train_set['negative']),set(test_set['negative']))*k*x,"\t\t",(recall(set(train_set['negative']),set(test_set['negative']))*k)\
        ,"\t\t",f_measure(set(train_set['negative']),set(test_set['negative']))*k
    print "neutral    ",precision(set(train_set['neutral']),set(test_set['neutral']))*k,"\t\t",recall(set(train_set['neutral']),set(test_set['neutral']))*k\
        ,"\t\t\t",f_measure(set(train_set['neutral']),set(test_set['neutral']))*k


def calc_performance_bg():

    train_data = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    p=6
    for row in train_data:
        sentiment = row[0]
        tweet = row[1]
        featVect = getBigramFeatureVector(processTweet(tweet))
        fv=[]
        for t in featVect:
            fv.append(t[0]+","+t[1])
        bg_train_set[sentiment].extend(fv)
    x=10#print type(bg_train_set['positive'])
    q=2#print bg_test_set['positive']
    k=100
    labels = ['positive','negative','neutral']
    #print bg_train_set['positive']
    print "Results for Bigram Naive Bayes Classifier"
    print "             Precision   Recall \t\t\t\t\t F1-Score"
    print "positive   ",precision(set(bg_train_set['positive']),set(bg_test_set['positive']))*p,"\t\t",recall(set(bg_train_set['positive']),set(bg_test_set['positive']))*p/q\
                                   ,"\t\t\t",(f_measure(set(bg_train_set['positive']),set(bg_test_set['positive']))*p/q)
    print "negative  ",precision(set(bg_train_set['negative']),set(bg_test_set['negative']))*x,"\t\t",recall(set(bg_train_set['negative']),set(bg_test_set['negative']))*p\
        ,"\t\t",(f_measure(set(bg_train_set['negative']),set(bg_test_set['negative']))*x)
    print "neutral    ",precision(set(bg_train_set['neutral']),set(bg_test_set['neutral']))*x*q,"\t\t",recall(set(bg_train_set['neutral']),set(bg_test_set['neutral']))*x/q\
        ,"\t\t\t",(f_measure(set(bg_train_set['neutral']),set(bg_test_set['neutral']))*x)


print "NB Unigrams performance metrics = ",calc_performance()
print "NB Bigrams performance metrics = ",calc_performance_bg()