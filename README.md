# Keyphrase - Find the most import key phrase from the corpus and use it for evaluating other document

Python - 2.7
# Dependencies
- NLTK
- Numpy
- Scikit-learn
- networkx

# Objective
- Generate the most import key-phrases from a corpus
- Using these key-phrases to compare other document
- Generating rank and score for these ke-phrases based on its influence on other document

# Methodology
 Bigram Collocation and Text rank used to identify the top key phrases
 Used Tfidf, cosine similarity, spearman metrics
 Two kinds of preprocessing. 
1) Normalized corpus removing special characters,numbers,punctuation,stopwords and word tokenized
2) Tokenized sentences. Sentences are fed into the training algorithm

# Output
- KEY PHRASES (using TextRank method). Shows rank, phrases and score
- KEY PHRASES (using Bigram-collocation method)
- Using the Bigram based top phrases, calculating the correlation/similarity between the documents
- Using the corpus as the training set, comparing each sentences in transcript for their similarity with corpus
- FINAL OUTPUT - Using the Top key phrases from the corpus(script) to score and rank transcript document

# Evaluating and Testing
Upload/add/change file name with path (if in other directory) with read_file function in the kp_eval.py
