import csv
from nltk import TweetTokenizer
from nltk.corpus import stopwords
import pickle

__author__ = 'Ritvika'
#import regex
import re
def processTweet(tweet):
    # process the tweets
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)',r'\1', tweet)
    #tweet = re.sub(r'(^[a-z]\s)|(\s[a-z]$)|(\s[a-z]\!$)|(\s[a-z]\?$)|(\s[a-z]\s)',' ',tweet)
    tweet = replaceTwoOrMore(tweet)
    #trim
    tweet = tweet.strip('\'"')
    #print tweet
    tt = TweetTokenizer()
    tweet_tokens = []
    try:
        tweet_tokens = tt.tokenize(tweet)
    except UnicodeDecodeError:
        #print "Error in tokenize"
        pass
    tweet_tokens = [ word.encode('utf-8') for word in tweet_tokens]
    #print tweet_tokens
    return tweet_tokens


#start process_tweet
def processTweetTrain(tweet):

    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)',r'\1', tweet)
    tweet = replaceTwoOrMore(tweet)
    tweet = tweet.strip('\'"')
    #print tweet
    tt = TweetTokenizer()
    tweet_tokens = []
    try:
        tweet_tokens = tt.tokenize(tweet)
    except UnicodeDecodeError:
        #print "Error in tokenize"
        pass
    tweet_tokens = [ word.encode('utf-8') for word in tweet_tokens]

        #print "tweet written"

   #return tweet_tokens

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end



#Read the tweets one by one and process it
def printProcessedTweet():
    """
    fp = open('sample_data.txt', 'r')
    line = fp.readline()

    print "printing processed tweet"
    while line:
        processedTweet = processTweet(line)
        print processedTweet
        line = fp.readline()
    fp.close()

    #end loop
        fp = csv.reader(open('TrainingDataSA.csv', 'r'), delimiter=',')
    for row in fp:
        #print row
        tweet = row[1]
    #processedTweet = str(processTweet(tweet))
    #print processedTweet
        processTweetTrain(tweet)
"""
#printProcessedTweet()
#print processTweet('Yeah! So excited ! Just updated my @Flipkart for #TheBigBillionDays.  https://t.co/cApJf5DCnC')
