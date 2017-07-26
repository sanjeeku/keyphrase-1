
# Evaluating the result. Training the model and printing out the results.

from kp_util import *
from kp_model import *

from kp_model import bigram_coll_score,bigram_coll,spearman_metrics,build_feature_matrix,keyphrases_score_by_textrank,bigram_coll_score
from kp_util import extract
from kp_model import compute_cosine_similarity

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

# =================================================================================================== 

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

# =================================================================================================== 

class top_phrase_transcript(object):
    def __init__(self,top_training_phrase,transcript):
        self.top_training_phrase = top_training_phrase
        self.transcript = transcript

    def transcript_scores(self):
        keyphrases = keyphrases_score_by_textrank(script)
        top_phrase_list = [word for (word,score) in keyphrases]
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0)
       
        tfidf_matrix =  tf.fit_transform(top_phrase_list)
        transcript_result = tf.transform(t3_sent)
        feature_names = tf.get_feature_names() 
        scores = zip(tf.get_feature_names(),
                     np.asarray(transcript_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for index,item in enumerate(sorted_scores):
            print index, " {0:50} Score: {1}".format(item[0], item[1])

# =================================================================================================== 

# MODEL EVALUATION
# Corpus and test files.You can change document here.Make sure your file path is the same path as code directory.
# script is the corpus/training document. And transcript are test files
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


# =================================================================================================== 

#You can SELECT the document/data to be fed for training or evaluation 
# List of best phrases, rank and score for a document(corpus or test), based on textrank.
c = phrase_score_textrank(script)

# List of best phrases, rank and score for a document(corpus or test), based on bigram collocation
p = phrase_score_collocation(s_normalized)


# Document correlation between corpus and a test document (or amongst the test documents)
# To evaluate correlation with other transcript - replace 'transcript_1' with 'transcript_2'/'transcript_3'
d = document_correlation(script,transcript_1)


# Document similarity [0-1] between a corpus and test document (or between the test documents). 
# Here, sentence to sentences are evaluated
# To evaluate similarity with other transcript - replace - 't3_sent' with 't1_sent'/'t2_sent'
s = document_similarity(s_sent,t3_sent)


# FINAL RESULT
#Taking the top phrase from the corpus, ranking and scoring transcript documents
# You can replace 't1_sent' with 't2_sent'/'t3_sent'
t = top_phrase_transcript(script,t1_sent)


####################### PRINT OUT THE RESULT ######################################################

"""KEY PHRASES (using TextRank method). Shows rank, phrases and score"""
c.print_result()


"""KEY PHRASES (using Bigram-collocation method)"""
p.print_result()


"""Using the Bigram based top phrases, calculating the correlation/similarity between the documents"""
d.print_result()


"""Using the corpus as the training set, comparing each sentences in transcript for their similarity with corpus"""
s.print_result()


"""FINAL OUTPUT - Using the Top key phrases from the corpus(script) to score and rank transcript document"""
t.transcript_scores()



# Run all the result at one go
def main():
    print "THE COMPLETE RESULT SET"
    print 
    c.print_result()
    p.print_result()
    d.print_result()
    s.print_result()
    t.transcript_scores()
    

main()






