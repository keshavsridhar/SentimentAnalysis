from pymongo import MongoClient as MC
import numpy
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import StratifiedKFold
import time

start=time.time()

def ExtractTweets(users,conn,dbname):
    #Enter server details below:
    client=MC(conn)
    db=client[dbname]
    #Candidates tweets to extract from MongoDB
    td=[]
    #Creating a dictionary to hold all the tweets from the presidential candidates
    for i in users:
        collection=db[i]
        #We care only for english tweets, hence the language filter
        td1=[[tweet['text'],i] for tweet in collection.find({"lang":"en"})]
        td.extend(td1)
    return td

def Preprocess(td):    
    rep=[]
    dem=[]
    for i in range(len(td)):
        td[i][0]=re.sub('\s?http(\w+|:)\W+.+','',td[i][0])
        td[i][0]=re.sub('@\w+','',td[i][0])
    rep=[td[i][0] for i in range(len(td)) if td[i][1] in ('realDonaldTrump','tedcruz','JohnKasich')]
    dem=[td[i][0] for i in range(len(td)) if td[i][1] not in ('realDonaldTrump','tedcruz','JohnKasich')]    

    train_rep_len=int(round(len(rep)*0.70))
    test_rep_len=int(len(rep)-train_rep_len)
    train_dem_len=int(round(len(dem)*0.70))
    test_dem_len=int(len(dem)-train_dem_len)
    
    train_data=[rep[i] for i in range(train_rep_len)] + [dem[i] for i in range(train_dem_len)]
    train_label=[0 for i in range(train_rep_len)] + [1 for i in range(train_dem_len)]
    test_data=[v for k,v in enumerate(rep) if k>=train_rep_len] + [v for k,v in enumerate(dem) if k>=train_dem_len]
    test_label=[0 for i in range(test_rep_len)] + [1 for i in range(test_dem_len)]
    
    train_label_vc=numpy.array(train_label)      
    test_label_vc=numpy.array(test_label)
    return train_data,test_data,train_label_vc,test_label_vc

def BOW(train_data):
    #print("Starting bag of words process:\n")
    Bag_of_words=[]
    sw=set(stopwords.words("english"))
    sw.add('rt')
    for tweet in train_data:
        Bag_of_words.extend(re.findall(r'\w+',tweet))
    Bag_of_words=[word.lower() for word in Bag_of_words]
    Bag_of_words=set(Bag_of_words)-sw
    Bag_of_words=list(Bag_of_words)
    #print("Bag of words process complete")
    return Bag_of_words

def vectorize_data(Bag_of_words,data):
    #print("Vectorizing data:\n")
    data_vector=[]
    for tweet in data:
        x=numpy.zeros(len(Bag_of_words))
        for word in set(re.findall(r'\w+',tweet)):
            if word.lower() in Bag_of_words:
                x[Bag_of_words.index(word.lower())]=1
        data_vector.append(x)
    #data_vector=[numpy.array(data_vector[i]) for i in range(len(data_vector))]
    return (data_vector)

def SVM_Classifier(Train_data_vc,Train_label,Test_data_vc,Test_label,start):
    print("Classifier being fitted\n")
    clf=OneVsRestClassifier(SVC(C=1,kernel='linear',gamma=1,verbose=False,probability=False))
    clf.fit(Train_data_vc,Train_label)
    print("\nClassifier fitted.\n")
    #predicted=cross_validation.cross_val_predict(clf,Train_data_vc,Train_label,cv=5)
    predicted=cross_validation.cross_val_predict(clf,Test_data_vc,Test_label,cv=5)
    
    print "Accuracy score:\n",metrics.accuracy_score(Test_label,predicted)
    print "Precision score:\n",metrics.precision_score(Test_label,predicted)
    print "Recall score:\n",metrics.recall_score(Test_label,predicted)
    print "Classification report:\n",metrics.classification_report(Test_label,predicted)
    print "Confusion_Marix:\n",metrics.confusion_matrix(Test_label,predicted)
    print("Total time:",(time.time()-start)/60)
    
    '''
    print "Accuracy score:\n",metrics.accuracy_score(Train_label,predicted)
    print "Precision score:\n",metrics.precision_score(Train_label,predicted)
    print "Recall score:\n",metrics.recall_score(Train_label,predicted)
    print "Classification report:\n",metrics.classification_report(Train_label,predicted)
    print "Confusion_Marix:\n",metrics.confusion_matrix(Train_label,predicted)
    print("Total time:",(time.time()-start)/60)
    '''

users=['HillaryClinton','JohnKasich','SenSanders','realDonaldTrump','tedcruz']
conn='mongodb://z604_final:lanif_406z@45.33.57.4:27017/tweetDB'
dbname="tweetDB"
tweetdict=ExtractTweets(users,conn,dbname)
train_data,test_data,train_label,test_label=Preprocess(tweetdict)
Bag_of_words=BOW(train_data)
train_data_vc=vectorize_data(Bag_of_words,train_data)
test_data_vc=vectorize_data(Bag_of_words,test_data)
SVM_Classifier(train_data_vc,train_label,test_data_vc,test_label,start)