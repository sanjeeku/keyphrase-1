
# Preprocess the document

import numpy as np
import nltk
import re
import string
import itertools

from nltk.metrics.spearman import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.collocations import *
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures



# Import and read the file
def read_file(filename):
    with open(filename,'rU') as scriptfile:
        script = (scriptfile.read()).decode('utf8')
    return script


class extract(object):
    def __init__(self, text):
        self.text = text

    def chunks(self, grammer=r'NP:{<DT>?<JJ>*<NN.*>+}'):
        import itertools, nltk, string, re
        all_chunked = []
        # exclude condidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        # tokenize, POS-tag and chunk using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammer)
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(self.text))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))

        # join constituent chunk words into a single chunked phrase
        candidates = [' '.join(word for word,pos,chunk in group).lower() for key,group in itertools.groupby(all_chunks, lambda(word,pos,chunk): chunk != 'O') if key]

        #return [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
        all_chunked.append(candidates)
        return all_chunked
        
    def words(self, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
        import itertools, nltk, string

        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))

        # tokenize and POS-tag words
        tagged_tokens = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(self.text)))

        # filter on certain POS tags and lowercase all words
        words = [word.lower() for word,tag in tagged_tokens if tag in good_tags and word.lower() not in stop_words and not all(char in punct for char in word)]

        return words    



def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_special_characters(text):
    #tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('',token) for token in text])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_numbers(tokenized_text):
    pattern = re.compile(r'[0-9]')
    filtered_token = [pattern.sub("",token) for token in tokenized_text]
    return filtered_token

def normalize_text(text):
    for c in string.punctuation:
        text = text.replace(c, "")
    text = remove_stopwords(text)
    text = tokenize_text(text)
    text = remove_special_characters(text)
    text = tokenize_text(text)
    text = remove_numbers(text)
    return text


def prep_sentences(corpus):
    sentences = [sent for sent in nltk.sent_tokenize(corpus)]
    sentences = remove_numbers(sentences)    
    return sentences


