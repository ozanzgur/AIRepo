to_import = ['utils']

from os.path import dirname, abspath
project_name = dirname(dirname(dirname(abspath(__file__)))).split('\\')[-1]

import sys
import importlib
for module in to_import:
    module_name = module.split('.')[-1]
    new_module = importlib.import_module(name = f'.{module}', package = project_name)
    sys.modules[__name__].__dict__.update({module_name: new_module})

################################################################################

import os
import logging
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import numpy as np
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
logger = logging.getLogger('pipeline')

class TextFeatureExtractor:
    @utils.catch('TEXTFEEATUREEXTRACTOR_INITERROR')
    def __init__(self, *args, **kwargs):
        # Initialize super
        
        # Load parameters
        def_args = dict()
        
        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
    
    """@utils.catch('TEXTFEEATUREEXTRACTOR_CALLERROR')
    def __call__(self, x, debug = False, **kwargs):
        assert(x is not None)
        if debug:
            logger.info(f'TEXTFEEATUREEXTRACTOR input: {str(x)}')

        return getattr(self, self.preprocessor_method)(x)"""

    @utils.catch('TEXTFEEATUREEXTRACTOR_PREDICTERROR')
    def transform(self, x):
        x.update({'output': _text2features(x['output'])})
        return x

    def _mask_numbers(self, text):
        # Replace each numeric char with #
        
        def repl(m):
            return f" {'#' * len(m.group())} "
        text = re.sub(r'[0-9]+', repl, text)
        return text

    @utils.catch('TEXTFEEATUREEXTRACTOR_PREPROCESSERROR')
    def fit_transform(self, x):
        """df must have columns : DOC_NAME, WORD, TARGET
        """
        df = x['data']
        df['TARGET'] = df['TARGET'].fillna('OTHER')
        df['WORD'] = df['WORD'].values.astype('U')
        df['WORD'] = df['WORD'].apply(lambda x: self._mask_numbers(x))

        #df['WORD'] = df['WORD'].apply(lambda x: _replace_province(x))
        #df = df.fillna(method='ffill')
        
        getter = DocGetter(df)
        docs = getter.docs

        X = [_doc2features(s) for s in docs]
        y = [_doc2labels(s) for s in docs]

        X_train = X[7:]
        X_val = X[:7]
        y_train = y[7:]
        y_val = y[:7]
        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=0)
        x['train_data'] = (X_train, y_train)
        x['val_data'] = (X_val, y_val)
        return x

class DocGetter(object):
    def __init__(self, data):
        self.n_doc = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, n_line, t) for w, t, n_line in zip(s['WORD'].values.tolist(), 
                                                                     s['TARGET'].values.tolist(),
                                                                     s['N_LINE'].values.tolist())]
        self.grouped = self.data.groupby('DOC_NAME').apply(agg_func)
        self.docs = [s for s in self.grouped]

def _replace_province(word):
    if word in province_names:
        return f'%PROVINCE%'
    else:
        return word

def _text2features(text):
    """Returns a list of examples.
    """
    words = wordpunct_tokenize(text)
    return [_word2features(words, i) for i in range(len(words))]

def _word2features(words, i):
    word = words[i][0]
    n_line = words[i][1]

    digit_count = sum(c=='#' for c in word)
    length = len(word)

    features = {
        'bias': 1.0,
        'word.index': i,
        'word.n_line': n_line
    }

    # If all digits
    if word.isdigit():
        features.update({
            'word.isdigit()': True,
            'word.digitcount': digit_count,
            'word.11digits()': digit_count == 11,
            'word.10digits()': digit_count == 10,
        })
    else: # Not all digit
        features.update({
            'word.digitratio': digit_count / length,
            'word.length': length,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.isProvinceName': word in province_names,
        })

    if i > 0:
        word1 = words[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper()
        })
        if i > 1:
            word2 = words[i-2][0]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.isupper()': word2.isupper()
            })
            if i > 2:
                word_other = words[i-3][0]
                features.update({
                    '-3:word.lower()': word_other.lower(),
                    '-3:word.isupper()': word_other.isupper()
                })
                if i > 3:
                    word_other = words[i-4][0]
                    features.update({
                        '-4:word.lower()': word_other.lower(),
                        '-4:word.isupper()': word_other.isupper()
                    })
    else:
        features['BOS'] = True
    if i < len(words)-1:
        word1 = words[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.isupper()': word1.isupper()
        })
        if i < len(words)-2:
            word2 = words[i+2][0]
            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.isupper()': word2.isupper()
            })
            if i < len(words)-3:
                word_other = words[i+3][0]
                features.update({
                    '+3:word.lower()': word_other.lower(),
                    '+3:word.isupper()': word_other.isupper()
                })
                if i < len(words)-4:
                    word_other = words[i+4][0]
                    features.update({
                        '+4:word.lower()': word_other.lower(),
                        '+4:word.isupper()': word_other.isupper()
                    })
        
    else:
        features['EOS'] = True
    return features

def _doc2features(doc):
    """Returns a list of examples.
    """
    words = [(ex[0], ex[1]) for ex in doc]
    return [_word2features(words, i) for i in range(len(words))]

def _doc2labels(doc):
    return [s[-1] for s in doc]
def _doc2tokens(doc):
    return [s[0] for s in doc]