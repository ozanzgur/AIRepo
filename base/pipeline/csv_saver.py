to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import logging
import pandas as pd
import numpy as np
from scipy import sparse
from os.path import join
logger = logging.getLogger('pipeline')

class CSVSaver:
    def __init__(self, **kwargs):
        def_args = dict(
            data_dir = '',
            shuffle = True,
            train_name = 'train',
            val_name = 'val',
            test_name = 'test',
            processed_extension = '.npz'
        )
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
            
        self.train_path = join(self.data_dir, self.train_name) + self.processed_extension
        self.val_path = join(self.data_dir, self.val_name) + self.processed_extension
        self.test_path = join(self.data_dir, self.test_name) + self.processed_extension
    
    def __call__(self, x, debug = False):
        self.train_data = x.get('train_data')
        self.val_data = x.get('val_data')
        self.test_data = x.get('test_data')
        
        """If datasets are DataFrames, y must be in column 'TARGET'.
        If datasets are sparse, datasets must be (x, y).
        
        Concatenates X and y and saves in a single file for each set.
        """
        logger.info(f'Data sizes:')
        try:
            logger.info(f'train: {self.train_data[0].shape}')
        except:
            logger.info('train data not found.')

        try:
            logger.info(f'val: {self.val_data[0].shape}')
        except:
            logger.info('val data not found.')
        try:
            logger.info(f'test: {self.test_data[0].shape}')
        except:
            logger.info('test data not found.')

        if self.processed_extension == '.csv':
            # Save to csv
            logger.info(f'Saving sets to csv:')
            
            # TRAIN
            logger.info(f'train: {self.train_path}')
            
            # Concatenate X and y
            train_data = self.train_data[0]
            train_data['TARGET'] = self.train_data[1]
            
            # Save as csv
            train_data.to_csv(self.train_path, index = False)
            
            
            # VAL
            logger.info(f'val: {self.val_path}')
            
            # Concatenate X and y
            val_data = self.val_data[0]
            val_data['TARGET'] = self.val_data[1]
            
            # Save as csv
            val_data.to_csv(self.val_path, index = False)
            
            # TEST
            logger.info(f'test: {self.test_path}')
            
            # Concatenate X and y
            test_data = self.test_data[0]
            test_data['TARGET'] = self.test_data[1]
            
            # Save as csv
            self.test_data.to_csv(self.test_path, index = False)
            
        elif self.processed_extension == '.npz':
            # Convert y to numpy array
            if isinstance(self.train_data[1], pd.Series):
                self.train_data[1] = self.train_data[1].to_numpy()
            if isinstance(self.val_data[1], pd.Series):
                self.val_data[1] = self.val_data[1].to_numpy()
            if isinstance(self.test_data[1], pd.Series):
                self.test_data[1] = self.test_data[1].to_numpy()
            
            # Save to npz (scipy sparse)
            logger.info(f'Saving sets to npz:')

            logger.info(f'train: {self.train_path}')
            train_data = [self.train_data[0], np.reshape(self.train_data[1], (-1,1))]
            sparse.save_npz(self.train_path, sparse.hstack(train_data))
            
            logger.info(f'val: {self.val_path}')
            val_data = [self.val_data[0], np.reshape(self.val_data[1], (-1,1))]
            sparse.save_npz(self.val_path, sparse.hstack(val_data))

            logger.info(f'test: {self.test_path}')
            test_data = [self.test_data[0], np.reshape(self.test_data[1], (-1,1))]
            sparse.save_npz(self.test_path, sparse.hstack(test_data))

        else:
            raise AttributeError(f'Wrong extension: {self.processed_extension}')
        
        self.input_size = self.train_data[0].shape[1]
        logger.info(f'Saved datasets.')