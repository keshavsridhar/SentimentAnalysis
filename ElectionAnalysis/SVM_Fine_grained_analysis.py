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
    print("Extraction time:\n",(time.time()-start)/60)    
    return td

def Preprocess(td):    
    hillary=[]
    bernie=[]
    kasich=[]
    trump=[]
    cruz=[]
    
    #Stripping links and user handles
    for i in range(len(td)):
        td[i][0]=re.sub('\s?http(\w+|:)\W+.+','',td[i][0])
        td[i][0]=re.sub('@\w+','',td[i][0])
    
    hillary=[td[i][0] for i in range(len(td)) if td[i][1]=="HillaryClinton"]
    bernie=[td[i][0] for i in range(len(td)) if td[i][1]=="SenSanders"]
    kasich=[td[i][0] for i in range(len(td)) if td[i][1]=="JohnKasich"]
    trump=[td[i][0] for i in range(len(td)) if td[i][1]=="realDonaldTrump"]
    cruz=[td[i][0] for i in range(len(td)) if td[i][1]=="tedcruz"]


    train_hillary_len=int(round(len(hillary)*0.70))
    train_bernie_len=int(round(len(bernie)*0.70))
    train_trump_len=int(round(len(trump)*0.70))
    train_kasich_len=int(round(len(kasich)*0.70))
    train_cruz_len=int(round(len(cruz)*0.70))
    
    test_hillary_len=len(hillary)-train_hillary_len
    test_bernie_len=len(bernie)-train_bernie_len
    test_trump_len=len(trump)-train_trump_len
    test_kasich_len=len(kasich)-train_kasich_len
    test_cruz_len=len(cruz)-train_cruz_len
    
    train_data=[hillary[i] for i in range(train_hillary_len)] + [bernie[i] for i in range(train_bernie_len)] + [trump[i] for i in range(train_trump_len)] + [kasich[i] for i in range(train_kasich_len)] + [cruz[i] for i in range(train_cruz_len)]
    train_label=[1 for i in range(train_hillary_len)] + [2 for i in range(train_bernie_len)] + [3 for i in range(train_trump_len)] + [4 for i in range(train_kasich_len)] + [5 for i in range(train_cruz_len)]
    test_data=[v for k,v in enumerate(hillary) if k>=train_hillary_len] + [v for k,v in enumerate(bernie) if k>=train_bernie_len] + [v for k,v in enumerate(trump) if k>=train_trump_len] + [v for k,v in enumerate(kasich) if k>=train_kasich_len] + [v for k,v in enumerate(cruz) if k>=train_cruz_len]
    test_label=[1 for i in range(test_hillary_len)] + [2 for i in range(test_bernie_len)] + [3 for i in range(test_trump_len)] + [4 for i in range(test_kasich_len)] + [5 for i in range(test_cruz_len)]
    
    train_label_vc=numpy.array(train_label)      
    test_label_vc=numpy.array(test_label)
    print("Data split time:\n",(time.time()-start)/60)
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
    print("Time taken to process bag of words:",(time.time()-start)/60)
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
    print("Vectorize time:",(time.time()-start)/60)
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
