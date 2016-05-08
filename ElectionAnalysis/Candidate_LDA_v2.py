import re
from nltk.corpus import stopwords
from gensim import corpora, models
from pprint import pprint


def ReadData(filename):
    speech = []
    # , encoding='utf-8'
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline()
        speech.append(line)
        for line in f:
            # c=f.readline()
            speech.append(line)
    return speech


def LDA(candidate):
    sw = set(stopwords.words("english"))
    sw.add("rt")

    for i in range(len(candidate)):
        x = re.findall(r"\w+'?[a-z]", candidate[i])
        x = set([word.lower() for word in x])
        candidate[i] = x - sw

    dictionary = corpora.Dictionary(candidate)
    corpus = [dictionary.doc2bow(text) for text in candidate]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=3)
    return ldamodel


file_name = 'Path/to/file'
speech = ReadData(file_name)
lda_result = LDA(speech)
pprint(lda_result.print_topics(num_topics=10))