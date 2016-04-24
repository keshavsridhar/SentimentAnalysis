from pymongo import MongoClient
import re
import nltk
import csv

def ExtractTweets(users):
    #Enter server details below:
    client=MongoClient('mongodb://z604_final:lanif_406z@45.33.57.4:27017/tweetDB')
    db=client["tweetDB"]
    #Candidates tweets to extract from MongoDB
    td={}
    #Creating a dictionary to hold all the tweets from the presidential candidates
    for i in users:
        collection=db[i]
        #We care only for english tweets, hence the language filter
        td[i]=[tweet for tweet in collection.find({"lang":"en"})]
    return td

#Applying simple regular expressions to just extract all the hashtags from the tweets
def ExtractHashtags(td):
    Hashtags={}
    for i in td.keys():
        Hashtags[i]=[]
        for j in range(len(td[i])):
            y=re.sub('\\n','',td[i][j]['text'])
            z=re.findall('#\w+',y)
            for x in z:
                Hashtags[i].append(x)
    return Hashtags

#Calculating the hashtags frequency over the whole set of tweets for each candidate separately
def CalcHashFreq(ht):
    hf={}
    for i in ht.keys():
        hf[i]=nltk.FreqDist(ht[i])
    return hf

if __name__=='__main__':
    tweetdict=ExtractTweets(['tweetStream'])
    hashtags=ExtractHashtags(tweetdict)
    hashfreq=CalcHashFreq(hashtags)
    #Writing out the frequencies to a file to be used later
    with open(r'C:\Users\keshs\Desktop\tweetstream.csv','w') as file:
        f=csv.writer(file)
        for i in hashfreq.keys():
            s=dict(hashfreq[i])
            file.write(i+'\n')
            for k,v in s.items():
                f.writerow([k,v])