from __future__ import division
import json
#import codecs
from nltk.text import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from math import log

stopwords=[x.encode('utf-8') for x in stopwords.words('english')]
tkzr=RegexpTokenizer(r'\w+')
f=open('Path\yelp_academic_dataset_review_preprocessed.json','r')
line=f.readline()
all_review_words=[]
pos_review_words=[]
neg_review_words=[]
pdict={}
ndict={}
combined_keys={}
p=0
n=0
#print line
i=0
while line:
    i=i+1
    line=f.readline()
    review=json.loads(line)
    #all_reviews.append(review['text'].encode('utf-8'))
    #review['text']=[word for word in tkzr.tokenize(review['text'].lower()) if word not in stopwords]
    review['text']=[word for word in tkzr.tokenize(review['text'].lower())]
    all_review_words=all_review_words+review['text']
    if i==10000:
        break
    if review['stars']>3:
        pos_review_words=pos_review_words+review['text']
        p=p+1
    elif review['stars']<3:
        neg_review_words=neg_review_words+review['text']
        n=n+1
    else:
        continue
f.close
pdist=FreqDist(pos_review_words)
ndist=FreqDist(neg_review_words)
#pdist=dict([(k,v) for k,v in pdist.items() if v>50])
#ndist=dict([(k,v) for k,v in ndist.items() if v>50])

'''
def PMI(dict1,c,a):
    for w in dict1.keys():
        try:
            p=(dict1[w]/len(c))/((a.count(w)/len(a))*len(c))
            pmi=log(p,2)
            dict1[w]=pmi
        except:
            continue
    print (dict1)
PMI(pdist,pos_review_words,all_review_words)
'''


def PMI(words,all_words,c,n):
    if c=="pos":
        for word in set(words):
            p=log(((words.count(word)/n)/(all_words.count(word)/9999)*n),2)
            #print(p)
            pdict[word]=p
    else:
        for word in set(words):
            p=log(((words.count(word)/n)/(all_words.count(word)/9999)*n),2)
            #print(p)
            ndict[word]=p

PMI(pos_review_words,all_review_words,"pos",p)
PMI(neg_review_words,all_review_words,"neg",n)

pdict_t=dict([(k,v) for k,v in pdict.items() if k not in stopwords and k in [key for key,val in pdist.items() if val>50]])
ndict_t=dict([(k,v) for k,v in ndict.items() if k not in stopwords and k in [key for key,val in ndist.items() if val>50]])

for key in pdict_t.keys():
    if key not in ndict_t.keys():
        continue
    else:
        combined_keys[key]=[pdict_t[key],ndict_t[key]]

pos_keys=dict([(k,v[0]) for k,v in combined_keys.items() if v[0]>v[1]])
neut_keys=dict([(k,v) for k,v in combined_keys.items() if v[1]==v[0]])
