import os
import re

import numpy as np
import underthesea
from gensim import corpora


SPECIAL_CHARS_REGEX = "[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]"
WORD_MIN_LENGTH = 2
WORD_MAX_LENGTH = 20


def load_corpus(data_dpath):
    corpus = list()
    for filename in os.listdir(data_dpath):
        fpath = os.path.join(data_dpath, filename)
        with open(fpath, "r", encoding="utf-8") as file:
            doc = file.read()
            corpus.append(doc) 
    return corpus

def clean_doc(doc):
    tokenized_doc = underthesea.word_tokenize(doc)
    non_special_chars_doc = map(lambda word: re.sub(SPECIAL_CHARS_REGEX, " ", word), 
                                tokenized_doc)                    
    processed_doc = [word.strip().lower().replace(" ", "_") 
                     for word in non_special_chars_doc
                         if  (len(word) >= WORD_MIN_LENGTH and
                              len(word) <= WORD_MAX_LENGTH)]
    return processed_doc

def clean_corpus(corpus):
    processed_corpus = list()
    for doc in corpus:
        processed_corpus.append(clean_doc(doc))
    return processed_corpus

def create_doc_term_matrix(corpus):
    processed_corpus = clean_corpus(corpus)
    dictionary = corpora.Dictionary(processed_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_corpus]
    return dictionary, doc_term_matrix

def get_keywords(dictionary, doc_term_matrix, model, n_keywords=7, doc_idx=-1):
    """Get keywords at doc_idx of corpus"""
    doc_term_value_vec = model.get_topics()[doc_idx]
    doc_term_idx_vec = [term_idx for term_idx, __ in doc_term_matrix[doc_idx]]
    # Remove terms that do not appear in the document
    for i in range(len(doc_term_value_vec)):
        if i not in doc_term_idx_vec:
            doc_term_value_vec[i] = -1
    # Get top n_keywords terms have highest values 
    top_n_indices = np.argpartition(doc_term_value_vec, -n_keywords)[-n_keywords:]
    keywords = [dictionary[i] for i in top_n_indices if doc_term_value_vec[i] > 0]
    return keywords
