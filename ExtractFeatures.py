import csv
import tokenize
import nltk

__author__ = 'Ritvika'
from PreProcessTweets import processTweet
from FeatureWords import *
tweets = []
featureList = []

#Read the tweets one by one and process it
def extractBulkFeatures():
    inpTweets = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    global tweets
    global featureList
    t = open('fkTweetsTS.txt','w')

    pt = open('fkProcessedTweets.txt','w')
    fv = open('fkFeatureVector.txt','w')
    fl = open('fkFeatureList.txt','w')

    for row in inpTweets:
        sentiment = row[0]
        tweet = row[1]
        processedTweet = processTweet(tweet)
        pt.write(" , ".join(processedTweet)+"\n")
        #pickle.dump(processedTweet,pt)
        featureVector = getFeatureVector(processedTweet)
        fv.write(" , ".join(featureVector)+"\n")
        #pickle.dump(featureVector,fv)
        #featureList.append(word for word in getFeatureVector(tweet))
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))
    #end loop
    #print tweets
    pt.close()
    fv.close()
    #t.write(tweets)
    featureList = list(set(featureList))
    #fl.write(" , ".join(featureList)+"\n")
    #print "Feature List :",featureList

    #pickle.dump(featureList,fl)
    fl.close()
#printFeatureVector()
#print "printin featureList \n \n"
#print featureList
def extractBulkBigramFeatures():
    inpTweets = csv.reader(open('TrainingDATA/flipkartTrainingData.csv', 'rb'), delimiter=',')
    global tweets
    global featureList
    t = open('fkBigramsTweetsTS.txt','w')
    featureList = []
    pt = open('fkBigramsProcessedTweets.txt','w')
    fv = open('fkBigramsFeatureVector.txt','w')
    fl = open('fkBigramsFeatureList.txt','w')

    for row in inpTweets:
        sentiment = row[0]
        tweet = row[1]
        processedTweet = processTweet(tweet)
        pt.write(" , ".join(processedTweet)+"\n")
        featureVector = getBigramFeatureVector(processedTweet)
        #print featureVector
        for lis in featureVector:
            fv.write("( "+lis[0]+" , "+lis[1]+" ) , ")
        fv.write("\n")
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))
    #print tweets
    #pt.close()
    #fv.close()
    #t.write(tweets)
    featureList = list(set(featureList))
    #print "BgFeature List :",featureList
    #for lis in featureList:
        #fl.write("( "+lis[0]+" , "+lis[1]+" ) , ")

    #pickle.dump(featureList,fl)
    #fl.close()
#printFeatureVector()
#print "printin featureList \n \n"
#print featureList
def extract_bigram_features(tweet):
    #print "extracting features of given tweet \n"
    features = {}
    for lis in featureList:
        features['contains(%s ,%s)' % (lis[0],lis[1])] = (lis in tweet)
    #print "features = ",features
    return features

def extract_features(tweet):
    #print "extracting features of given tweet \n"
    tweet_words = (re.split(r'[\s\'",.!?]\s*',' '.join(tweet)))
    #print "tweet-words : \n", tweet_words
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    #print "features = ",features
    return features
#end
#s_tweet = 'Yeah! So excited ! Just updated my @Flipkart for #TheBigBillionDays.  https://t.co/cApJf5DCnC'
#print extract_features(list(s_tweet.lower()))
#extractBulkFeatures()
#print extract_features(getFeatureVector(processTweet('Yeah! So excited ! Just updated my @Flipkart for #TheBigBillionDays.  https://t.co/cApJf5DCnC')))

#extractBulkBigramFeatures()
#print extract_bigram_features(getBigramFeatureVector(processTweet('Yeah! So excited ! Just updated my @Flipkart for #TheBigBillionDays.  https://t.co/cApJf5DCnC')))


