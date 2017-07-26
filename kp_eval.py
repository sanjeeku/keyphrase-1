
# coding: utf-8

# In[28]:

from kp_util import *
from kp_model import *


# In[29]:

from kp_model import bigram_coll_score,bigram_coll,spearman_metrics,build_feature_matrix,keyphrases_score_by_textrank,bigram_coll_score,compute_cosine_similarity
from kp_util import extract


# In[30]:

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


# In[31]:

class phrase_score_textrank(object):
    def __init__(self,corpus):
        self.corpus = corpus
    
    def print_result(self):
        print "KEY PHRASE BY TEXTRANK"
        print "="*40
        print "Total keyphrase idendified  :", len(keyphrases_score_by_textrank(self.corpus))
        print "="*40
        for index, (word, score) in enumerate(keyphrases_score_by_textrank(self.corpus)):
            print '{}  {} : {}'.format(index, word, score)
        print "="*20+"End"+"="*20

# ===================================================================================================        
        
class document_correlation(object):
    def __init__(self,corpus,transcript):
        self.corpus = corpus
        self.transcript = transcript
    
    def print_result(self):        
        print "-"*40
        print "DOCUMENT CORRELATION"
        print "="*40
        print            
        document_correlation = spearman_metrics(self.corpus,self.transcript)
        print "The correlation between the two document:".format(self.corpus, self.transcript), document_correlation


# ===================================================================================================          
        
class phrase_score_collocation(object):
    def __init__(self,normalized_corpus):
        self.normalized_corpus = normalized_corpus
            
    def print_result(self):
        print "-"*40
        print "KEY PHRASE COMPARISON USING BIGRAM-COLLOCATION"
        print "="*40
        print "Total keyphrase idendified  :", len(bigram_coll_score(self.normalized_corpus))
        print
        
        for index, ((word1,word2), score) in enumerate(bigram_coll_score(self.normalized_corpus)):
            print index, word1,word2, score


# In[32]:

class document_similarity(object):
    def __init__(self, corpus,transcript):
        self.corpus = corpus
        self.transcript = transcript

        self.tfidf_vectorizer, self.tfidf_features = build_feature_matrix(self.corpus,feature_type='tfidf', ngram_range=(1,1),min_df=0,max_df=1.0)
        self.query_doc_tfidf = self.tfidf_vectorizer.transform(self.transcript)

    def print_result(self):
        
        print "Document similarity result for our script and transcript document"
        print '='*80
        for index, doc in enumerate(self.transcript):
            doc_tfidf = self.query_doc_tfidf[index]
            top_similar_docs = compute_cosine_similarity(doc_tfidf, self.tfidf_features)
            print 'Transcript', index+1, ':', doc
            print '#Top', len(top_similar_docs), 'similar_doc:'
            print 
            print '#'*80
            print 'Similarity with Corpus'
            for doc_index, sim_score in top_similar_docs:
                print "Corpus index: ",doc_index+1 
                print "Similarity score: ", sim_score 
                print "Corpus Sentence : ", self.corpus[doc_index]
                print '-'*80
            print 


# In[33]:

# print result
# Corpus and test files.You can change files here
# script is the corpus/training file. And transcript are test files
script = read_file('script.txt')
transcript_1 = read_file('transcript_1.txt')
transcript_2 = read_file('transcript_2.txt')
transcript_3 = read_file('transcript_3.txt')


# Normalized and word tokenized
s_normalized = normalize_text(script)
t1_normalized = normalize_text(transcript_1)
t2_normalized = normalize_text(transcript_2)
t3_normalized = normalize_text(transcript_3)

# Sent tokenized
s_sent = prep_sentences(transcript_1)
t1_sent = prep_sentences(transcript_1)
t2_sent = prep_sentences(transcript_2)
t3_sent = prep_sentences(transcript_3)



# In[34]:

#You can change the document/data to be fed for training or evaluation 
# List of best phrase, rank and score for a document(corpus or test), based on text rank.
c = phrase_score_textrank(script)

# List of best phrase, rank and score for a document(corpus or test), based on bigram collocation
p = phrase_score_collocation(s_normalized)


# document correlation between corpus and a test document (or between the test documents)
d = document_correlation(script,transcript_1)


# Document similarity [0-1] between a corpus and test document (or between the test documents). 
# Here, sentence to sentences are evaluated
s = document_similarity(s_sent,t3_sent)


# In[ ]:




# In[35]:

phrase_score_textrank(script).print_result()


# In[36]:

d.print_result()


# In[37]:

p.print_result()


# In[38]:

s.print_result()


# In[ ]:




# In[ ]:



