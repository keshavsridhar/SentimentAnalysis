import re
from pymongo import MongoClient

def ExtractTweets(collection_name):
    #Enter server details below:
    client=MongoClient('') # Connection String for the database
    db=client["tweetDB"]
    collection = db[collection_name]
    return [tweet for tweet in collection.find({"lang":"en"})]

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
