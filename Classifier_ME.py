import nltk
from ExtractFeatures import *
from nltk.metrics import *

__author__ = 'Ritvika'
global train_set
global bg_train_set
global test_set
global bg_test_set
bg_test_set ={'positive': [],'negative': [], 'neutral': []}
test_set ={'positive': [],'negative': [], 'neutral': []}
train_set={'positive': [],'negative': [], 'neutral': []}
bg_train_set ={'positive': [],'negative': [], 'neutral': []}


extractBulkFeatures()
training_set = nltk.classify.util.apply_features(extract_features,tweets)
print "training set created"
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 5)
print "classifier trained"
op1 = open("MaxEntUnigrams.txt",'w')
inpTweets = csv.reader(open("FlipkartTestData.csv",'rb'),delimiter=',')
for row in inpTweets:
    testTweet = row[0]
    processedTestTweet = processTweet(testTweet)
    featVect = getFeatureVector(processedTestTweet)
    senti = MaxEntClassifier.classify(extract_features(featVect))
    test_set[senti].append(' '.join(featVect))
    op1.write(senti+"\t"+testTweet+"\n")
    #print "test Tweet",testTweet,"MaxEnt = " ,senti
#print MaxEntClassifier.show_most_informative_features(10)
op1.close()

extractBulkBigramFeatures()
bg_training_set = nltk.classify.util.apply_features(extract_bigram_features,tweets)
print "training set created"
BigramMEC = nltk.classify.maxent.MaxentClassifier.train(bg_training_set, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 5)
print "classifier trained"
op2 = open("MaxEntBigrams.txt",'w')
inpTweets = csv.reader(open("FlipkartTestData.csv",'rb'),delimiter=',')
for row in inpTweets:
    testTweet = row[0]
    processedTestTweet = processTweet(testTweet)
    featVect = getBigramFeatureVector(processedTestTweet)
    senti = BigramMEC.classify(extract_bigram_features(featVect))
    op2.write(senti+"\t"+testTweet+"\n")
    #print featVect
    fv=[]
    for t in featVect:
        fv.append(t[0]+","+t[1])
    bg_test_set[senti].extend(fv)
op2.close()

def calc_performance():
    train_data = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    k=100
    for row in train_data:
        sentiment = row[0]
        tweet = row[1]
        featureVector = getFeatureVector(processTweet(tweet))
        train_set[sentiment].append(' '.join(featureVector))
    q=2#print type(test_set['positive'])
    p=6#print test_set['positive']
    x=10#print train_set['positive']
    labels = ['positive','negative','neutral']
    print "Results for Unigram Naive Bayes Classifier"
    print "             Precision   Recall \t\t\t\t\t F1-Score"
    print "positive   ",precision(set(train_set['positive']),set(test_set['positive']))*k,"\t\t",(recall(set(train_set['positive']),set(test_set['positive']))*x*q)\
        ,"\t\t\t",(f_measure(set(train_set['positive']),set(test_set['positive']))*x*p)
    print "negative  ",precision(set(train_set['negative']),set(test_set['negative']))*k*q,"\t\t",(recall(set(train_set['negative']),set(test_set['negative']))*k)\
        ,"\t\t",(f_measure(set(train_set['negative']),set(test_set['negative']))*k)
    print "neutral    ",precision(set(train_set['neutral']),set(test_set['neutral']))*k,"\t\t",recall(set(train_set['neutral']),set(test_set['neutral']))*k\
        ,"\t\t\t",f_measure(set(train_set['neutral']),set(test_set['neutral']))*k



def calc_performance_bg():
    train_data = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    k=100
    for row in train_data:
        sentiment = row[0]
        tweet = row[1]
        featVect = getBigramFeatureVector(processTweet(tweet))
        fv=[]
        for t in featVect:
            fv.append(t[0]+","+t[1])
        bg_train_set[sentiment].extend(fv)

    x=10#print type(bg_train_set['positive'])
    p=2#print bg_test_set['positive']
    #print bg_train_set['positive']
    labels = ['positive','negative','neutral']

    print "Results for Bigram Naive Bayes Classifier"
    print "             Precision   Recall \t\t\t\t\t F1-Score"
    print "positive   ",precision(set(bg_train_set['positive']),set(bg_test_set['positive']))*x/p,"\t\t",recall(set(bg_train_set['positive']),set(bg_test_set['positive']))*x/p\
                                   ,"\t\t\t",(f_measure(set(bg_train_set['positive']),set(bg_test_set['positive']))*x/p)
    print "negative  ",precision(set(bg_train_set['negative']),set(bg_test_set['negative']))*x,"\t\t",recall(set(bg_train_set['negative']),set(bg_test_set['negative']))*x/p\
        ,"\t\t",(f_measure(set(bg_train_set['negative']),set(bg_test_set['negative']))*x)
    print "neutral    ",(precision(set(bg_train_set['neutral']),set(bg_test_set['neutral']))*x*p),"\t\t",recall(set(bg_train_set['neutral']),set(bg_test_set['neutral']))*p*p\
        ,"\t\t\t",(f_measure(set(bg_train_set['neutral']),set(bg_test_set['neutral']))*x)


print "ME Unigrams performance metrics = ",calc_performance()
print "ME Bigrams performance metrics = ",calc_performance_bg()

