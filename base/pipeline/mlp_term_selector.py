to_import = [
    'utils',
    'base.model.multi_model'
]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import logging
import pandas as pd
import os
logger = logging.getLogger('pipeline')

class MLPTermSelector:
    def __init__(self, **kwargs):
        def_args = dict(
            term_selector_limit = 1000,
            data_dir = ''
        )
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        
        self._load_vocabulary()
        
        if self.vocabulary is None:
            logging.warning('vocabulary.csv could not be found, you must pass "features" dataframe from prev. steps')

        
    
    def __call__(self, x, debug = False):        
        logger.info('Fitting MLP...')
        if 'features' in x:
            self.vocabulary = x['features']

        logger.info(f'MLPTERMSELECTOR: previous n_features: {len(self.vocabulary)}')
        self.multi_model_selector = multi_model.MultiModel(
            model_type = 'mlp',
            model_method = 'fit',
            input_size = len(self.vocabulary)
        )

        self.multi_model_selector.fit(x)

        features = self.vocabulary['term'].tolist()
        
        # Get N most important terms
        logger.info('Getting feature importances...')
        terms = self.multi_model_selector.model.feature_importances(
            feature_list = features,
            limit = self.term_selector_limit,
            plot = 50)
        
        term_idx = terms.index.tolist()
        
        # Save new vocabulary
        self.vocabulary = self.vocabulary.iloc[term_idx]
        logger.info(f'Vocabulary size: {len(self.vocabulary)}')
        
        self._save_vocabulary()

        x['features'] = self.vocabulary
        return x
    
    def _load_vocabulary(self):
        try:
            logger.info('Loading vocabulary...')
            self.vocabulary = pd.read_csv(os.path.join(self.data_dir,'vocabulary.csv'))
            logger.info(f'Term count in vocabulary: {len(self.vocabulary)}')
        except FileNotFoundError:
            logger.info('Vocabulary could not be found.')
            self.vocabulary = None
    
    def _save_vocabulary(self):
        logger.info('Saving vocabulary...')
        self.vocabulary.to_csv(os.path.join(self.data_dir,'vocabulary.csv'), index = False)
        self.n_terms = len(self.vocabulary)
        logger.info(f'Term count in vocabulary: {self.n_terms}')