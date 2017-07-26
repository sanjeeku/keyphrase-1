
# coding: utf-8

# In[88]:

from kp_util import *


# In[89]:

import nltk
import re
import string
import itertools
import numpy as np
from nltk.metrics.spearman import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.collocations import *
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

from itertools import takewhile, tee, izip
import networkx


# In[ ]:

# Corpus and test files
script = read_file('script.txt')
transcript_1 = read_file('transcript_1.txt')
transcript_2 = read_file('transcript_2.txt')
transcript_3 = read_file('transcript_3.txt')


# Normalized and word tokenized
s = normalize_text(script)
t1 = normalize_text(transcript_1)
t2 = normalize_text(transcript_2)
t3 = normalize_text(transcript_3)

# Sent tokenized
script_sent = prep_sentences(transcript_1)
transcript_1_sent = prep_sentences(transcript_1)
transcript_2_sent = prep_sentences(transcript_2)
transcript_3_sent = prep_sentences(transcript_3)


# In[ ]:




# In[91]:

bigram_measure = BigramAssocMeasures()
def bigram_coll(text,n=50):
    bigram_measure = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(text,2)
    finder.apply_freq_filter(1)
    return finder.nbest(bigram_measure.likelihood_ratio,n)

def bigram_coll_score(text, n=500):
    bigram_measure = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents([text])
    finder.apply_freq_filter(2)
    scored = finder.score_ngrams(bigram_measure.likelihood_ratio)
    return scored[:n]


# In[92]:


def spearman_metrics(text1, text2):
    text1_list = bigram_coll(text1)
    text2_list = bigram_coll(text2)
    rank1 = (list(ranks_from_sequence(text1_list)))
    rank2 = (list(ranks_from_sequence(text2_list)))
    comparison = spearman_correlation(rank1, rank2)
    return comparison


# In[93]:

def build_feature_matrix(documents, feature_type='frequency', ngram_range=(1,1), min_df =0.0, max_df=1):
    feature_type = feature_type.lower().strip()
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    else:
        raise Exception("Wrong feature type entered. Possible values 'binary', 'frequency', 'tfidf'")
        
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


# In[94]:

def keyphrases_score_by_textrank(text, n_keywords=0.05):
    # tokenize
    words = [word.lower()
            for sent in nltk.sent_tokenize(text)
            for word in nltk.word_tokenize(sent)]
    
    candidates = extract(text).words()
    
    # build graph, each node is a unique condidates
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """ s-> (s0,s1), (s1,s2),(s2,s3)...."""
        a, b = tee(iterable)
        next(b,None)
        return izip(a,b)
    for w1,w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1,w2]))
    
    # score nodes using default pagerank algoritm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates)*n_keywords))
    
    word_ranks = {word_rank[0]: word_rank[1] for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1],reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i<j :
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words)/float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
            
    return sorted(keyphrases.iteritems(), key=lambda x:x[1], reverse=True)


# In[ ]:




# In[53]:




# In[12]:




# In[59]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



