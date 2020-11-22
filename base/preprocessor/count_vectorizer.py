to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import os
import logging
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
logger = logging.getLogger('pipeline')

class CV:
    @utils.catch('CV_INITERROR')
    def __init__(self, step_name, *args, **kwargs):
        # Initialize super
        
        # Load parameters
        def_args = dict(
            ngram_range = (1,1),
            vocabulary = None,
            data_dir = '',
            word_count_limit = 1000
        )
        
        self.CV = None
        self.step_name = step_name
        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        
        ########################################################################
        # Vocabulary
        
        self._load_vocabulary()
        
        if self.vocabulary is not None:
            self.n_term = len(self.vocabulary)
            self._create_cv()
            logger.info(f'Term count in vocabulary: {self.n_term}')
        else:
            logger.info('Vocabulary not loaded.')
    
    ############################################################################
    # Default methods
    
    """@utils.catch('CV_CALLERROR')
    def __call__(self, x, debug = False, **kwargs):
        assert(x is not None)
        if debug:
            logger.info(f'CV input: {str(x)}')

        return getattr(self, self.preprocessor_method)(x)"""

    @utils.catch('CV_PREDICTERROR')  
    def transform(self, x, debug = False):
        assert(self.vocabulary is not None)
        
        x.update({'output': self.CV.transform(x['output'])})
        return x

    @utils.catch('CV_PREPROCESSERROR')
    def fit_transform(self, x, debug = False):
        df = x['data']

        y = df['TARGET']
        
        # Fit count vectorizer
        if self.CV is None:
            self.vocabulary = x.get('features')
            self._create_cv()

        self.CV.fit(df['TEXT'].tolist())
        
        # Get word counts for csv
        logger.info('Getting word counts...')
        word_count_vector = self.CV.transform(df['TEXT'].tolist())

        # Eliminate words with low frequency

        if not 'features' in x:
            # Build a dataframe with words
            logger.info('Creating new vocabulary...')
            self.vocabulary = pd.DataFrame({
                'freq': np.array(word_count_vector.sum(axis = 0)).squeeze(),
                'term': self.CV.get_feature_names()
            })
            
            logger.info(f'Removing words with count < {self.word_count_limit}')
            term_idx_to_keep = self.vocabulary[self.vocabulary['freq'] > self.word_count_limit].index
            self.vocabulary = self.vocabulary.iloc[term_idx_to_keep]

            # Slice word count vector with the words to keep
            word_count_vector = sparse.lil_matrix(
            sparse.csr_matrix(word_count_vector)[:,term_idx_to_keep])

        logger.info(f'Unique term count: {len(self.vocabulary)}')

        self._save()
        x['data'] = (word_count_vector, y)
        x['features'] = self.vocabulary
        return x

    @utils.catch('CV_SAVEERROR')
    def _save(self):
        self._save_vocabulary()
    
    ############################################################################
    # Other methods
        
    @utils.catch('CV_CREATECOUNTVECTORIZERERROR')
    def _create_cv(self):
        if self.vocabulary is None:
            logger.error('Vocabulary does not exist.')
        
        logger.info('Creating count vectorizer...')
        self.CV = CountVectorizer(vocabulary = self.vocabulary['term'].tolist() \
                                    if self.vocabulary is not None else None,
                                  ngram_range = self.ngram_range)
    
    @utils.catch('CV_SAVEVOCABULARYERROR')
    def _save_vocabulary(self):
        logger.info('Saving vocabulary...')
        self.vocabulary.to_csv(os.path.join(self.data_dir,'vocabulary.csv'), index = False)
        self.n_terms = len(self.vocabulary)
        logger.info(f'Term count in vocabulary: {self.n_terms}')
        
            
    @utils.catch('CV_LOADVOCABULERYERROR')
    def _load_vocabulary(self):
        try:
            logger.info('Loading vocabulary...')
            self.vocabulary = pd.read_csv(os.path.join(self.data_dir,'vocabulary.csv'))
        except FileNotFoundError:
            logger.info('Vocabulary could not be found.')
            self.vocabulary = None
            
    @utils.catch('CV_GETFEATURESERROR')
    def get_features(self):
        return self.vocabulary['term'].tolist()