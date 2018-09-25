import csv
from nltk import bigrams
from nltk.corpus import stopwords
import re
from  PreProcessTweets import *
import pickle

__author__ = 'Ritvika'

#start getfeatureVector

def getFeatureVector(tweet_tokens):
    #print " \t getFeatureVector called "
    featureVector = []
    for w in tweet_tokens:
        #replace two or more with two occurrences
        #w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopwords.words('english') or w == 'AT_USER' or w == 'URL' or re.search(r'^[a-z]$',w) or  val is None):
            continue
        else:
            featureVector.append(w.lower())
    #print type(featureVector)
    return featureVector
#end
def getBigramFeatureVector(tweet_tokens):
    #print " \t getFeatureVector called "
    temp = []
    for w in tweet_tokens:
        w = w.strip('\'"!?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(w in stopwords.words('english') or w == 'AT_USER' or w == 'URL' or re.search(r'^[a-z]$',w) or  val is None):
            continue
        else:
            temp.append(w.lower())
    featureVector = list(bigrams(temp))
    return featureVector

def getFeatureVectorTrain():
    #print " \t getFeatureVector called "
    featureVector = []
    print "inside getFeatureVectorTrain \n"
    with open("ProcessedTweetTokens.txt",'r') as pt:
        fp = open('FeatureVectors.txt','a')
        print "ProcessedTweetTokens file opened \n"
        tweet_tokens = pickle.load(pt)
        for w in tweet_tokens:
            w = w.strip('\'"?,.')
            #check if the word stats with an alphabet
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
             #ignore if it is a stop word
            if(w in stopwords.words('english') or w == 'AT_USER' or w == 'URL' or re.search(r'^[a-z]$',w) or  val is None):
                continue
            else:
                featureVector.append(w.lower())
            #print type(featureVector)
        pickle.dump(featureVector,fp)
        print "feature vector written \n"

#Read the tweets one by one and process it

#getFeatureVectorTrain()


def printFeatureVector():
    """
    fp = open('sample_data.txt', 'r')
    line = fp.readline()
    print "printing feature vector"
    while line:
        processedTweet = processTweet(line)
        featureVector = getFeatureVector(processedTweet)
        print featureVector
        line = fp.readline()
    #end loop
    """
    #fp.close()
    """
    fp = csv.reader(open('corpus.csv', 'r'), delimiter=',')
    for row in fp:
        tweet = row[1]
        processedTweet = processTweet(tweet)
        featureVector = getFeatureVector(processedTweet)
        print featureVector
"""
#print getBigramFeatureVector(processTweet('Yeah! So excited ! Just updated my @Flipkart for #TheBigBillionDays.  https://t.co/cApJf5DCnC'))
#printFeatureVector()
