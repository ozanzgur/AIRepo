to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import logging
import pandas as pd
import numpy as np
logger = logging.getLogger('pipeline')

class DataSplitter:
    @utils.catch('DATASPLITTER_INITERROR')
    def __init__(self, **kwargs):
        # TODO: Add shuffling in splitter
        def_args = dict(
            train_ratio = 0.75,
            val_ratio = 0.125,
            #shuffle = True,
        )
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
                
    @utils.catch('DATASPLITTER_CALLERROR')
    def __call__(self, x, debug = False):
        """Split into train, val, test.
        """
        assert(x is not None)
        data = x['data']
        train_data = None
        val_data = None
        test_data = None
        isdataframe = None
        
        # Find example counts for each set
        self.n_examples = data[0].shape[0]
        self.n_train = int(self.n_examples * self.train_ratio)
        self.n_val = int(self.n_examples * self.val_ratio)
        self.n_test = self.n_examples - self.n_train - self.n_val
        
        logger.info(f'Set sizes:')
        logger.info(f'train: {self.n_train}')
        logger.info(f'val: {self.n_val}')
        logger.info(f'test: {self.n_test}')
        if self.n_test < 0:
            raise ValueError('Train + validation ratios must be < 1')

        # 4 - Separate data into train, test, val
        if isinstance(data[0], pd.DataFrame):
            logger.info('Dataset is in a dataframe.')
            isdataframe = True

            train_data = [data[0].iloc[:self.n_train],
                               data[1].iloc[:self.n_train]]
            
            val_data = [data[0].iloc[self.n_train:self.n_val + self.n_train],
                             data[1].iloc[self.n_train:self.n_val + self.n_train]]
            
            test_data = [data[0].iloc[self.n_val + self.n_train:],
                              data[1].iloc[self.n_val + self.n_train:]]
            logger.info('Data was split into train, val, test.')
        else:
            isdataframe = False
            logger.info('Dataset is in a numpy array.')
            
            # If datasets are numpy array or sparse
            train_data = [data[0][:self.n_train],
                               data[1][:self.n_train]]
            
            val_data = [data[0][self.n_train:self.n_val + self.n_train],
                             data[1][self.n_train:self.n_val + self.n_train]]
            
            test_data = [data[0][self.n_val + self.n_train:],
                              data[1][self.n_val + self.n_train:]]
            logger.info('Data was split into train, val, test.')
            
        assert(self.n_train == train_data[0].shape[0])
        assert(self.n_val == val_data[0].shape[0])
        assert(self.n_test == test_data[0].shape[0])
        
        x['train_data'] = train_data
        x['val_data'] = val_data
        x['test_data'] = test_data

        return x