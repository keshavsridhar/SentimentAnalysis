from pymongo import MongoClient as MC
from nltk.corpus import stopwords
from gensim import corpora,models
import numpy
import re
import time

start=time.time()

def ExtractTweets(user,conn,dbname):
    #Enter server details below:
    client=MC(conn)
    db=client[dbname]
    #Candidates tweets to extract from MongoDB
    td=[]
    #Creating a dictionary to hold all the tweets from the presidential candidates
    collection=db[user]
    #We care only for english tweets, hence the language filter
    td1=[tweet['text'] for tweet in collection.find({"lang":"en"})]
    td.extend(td1)
    return td

sw=set(stopwords.words("english"))
sw.add("rt")
user_des=ExtractTweets('tweetStream','conn_string','tweetDB')
for i in range(len(user_des)):
    user_des[i]=re.sub('\s?http(\w+|:)\W+.+','',user_des[i])
    user_des[i]=re.sub('@\w+','',user_des[i])

for i in range(len(user_des)):
    x=re.findall(r"\w+'?[a-z]",user_des[i])
    x=set([word.lower() for word in x])
    user_des[i]=x-sw
    
dictionary=corpora.Dictionary(user_des)
corpus=[dictionary.doc2bow(text) for text in user_des]
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary)
