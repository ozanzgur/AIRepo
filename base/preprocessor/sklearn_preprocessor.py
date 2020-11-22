to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import importlib
import logging
from scipy import sparse
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
logger = logging.getLogger('pipeline')

class SklearnPreprocessor:
    def __init__(self, step_name, **kwargs):
        assert('sklearn_class' in kwargs)
        
        # Get parameters
        preprocessor_kwargs = kwargs.copy()
        preprocessor_kwargs.pop('model_method', None)
        self.data_dir = preprocessor_kwargs.pop('data_dir')
        sklearn_class = preprocessor_kwargs.pop('sklearn_class')
        self.sklearn_class_name = sklearn_class.split('.')[-1]
        self.step_name = step_name

        # Import sklearn module and get class
        sklearn_module_name = sklearn_class[:-len(self.sklearn_class_name) -1]

        logger.info(f'Importing module: {sklearn_module_name}')
        sklearn_module = importlib.import_module(sklearn_module_name)

        logger.info(f'Getting class: {self.sklearn_class_name}')
        self.sklearn_class = getattr(sklearn_module, self.sklearn_class_name)
        self.preprocessor_kwargs = preprocessor_kwargs
        self.preprocessor = None
    
    ############################################################################
    # Default methods
    
    def __call__(self, x, debug = False, **kwargs):
        assert(x is not None)
        if debug:
            logger.info(f'{self.step_name} input: {str(x)}')

        return getattr(self, self.preprocessor_method)(x)
    
    def transform(self, x, debug = False):
        if self.preprocessor is None:
            self._load() #self.preprocessor = self.sklearn_class(**self.preprocessor_kwargs)
        
        x.update({'output': self.preprocessor.transform(x['output'])})
        return x

    def fit_transform(self, x, debug = False):
        self.preprocessor = self.sklearn_class(**self.preprocessor_kwargs)
        
        x_data = x['data'][0]
        y_data = x['data'][1]
        
        x_data = self.preprocessor.fit_transform(x_data)
        
        self._save()
        x['data'] = (x_data, y_data)
        return x

    def _save(self):
        if self.preprocessor is not None:
            logger.info(f'Saving {self.step_name}...')
            with open(os.path.join(self.data_dir, self.step_name), 'wb') as fp:
                pickle.dump(self.preprocessor, fp)
        else:
            logger.error(f'{self.step_name} class does not exist.')
    
    def _load(self):
        try:
            logger.info(f'Loading {self.step_name}...')
            with open (os.path.join(self.data_dir, self.step_name), 'rb') as fp:
                self.preprocessor = pickle.load(fp)
                
        except FileNotFoundError:
            logger.error(f'{self.step_name} could not be found.')
            

