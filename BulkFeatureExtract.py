import nltk
import csv
#from ExtractFeatures import extract_features
from FeatureWords import getFeatureVector
from PreProcessTweets import processTweet

__author__ = 'Ritvika'
inpTweets = csv.reader(open('TrainingDataSA.csv', 'rb'), delimiter=',')
featureList = []

# Get tweet words
tweets = []

for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))
print featureList
# Extract feature vector for all tweets in one shote
#training_set = nltk.classify.util.apply_features(extract_features, tweets)
#print tweets
#NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
