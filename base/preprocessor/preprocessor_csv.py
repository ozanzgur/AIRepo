to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################
    
import pandas as pd
import numpy as np
import os

import logging
logger = logging.getLogger('pipeline')
from scipy import sparse

class BasePreprocessorCSV:
    @utils.catch('BASEPREPROCESSOR_INITERROR')
    def __init__(self, **kwargs):
        def_args = dict(
            raw_name = 'raw_data.csv',
            data_dir = '',
            load_data = True,
            train_ratio = 0.75,
            val_ratio = 0.125,
            #classes = None,
            save_sets = True,
            shuffle = True,
            train_name = 'train',
            val_name = 'val',
            test_name = 'test',
            processed_extension = '.npz',
            word_count_limit = 5,
            read_params = dict(
            )
        )
        
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})

        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.n_examples = None
        self.n_train = None
        self.n_val = None
        self.n_test = None
        self.batch_size = None
        self.load_raw = False
        self.loaded_preprocessed = False
        
        # Extension must be .csv or .npz
        assert(self.processed_extension in ['.csv', '.npz'])
        
        # Load raw data. If preprocessed version exists, load that one.
        if not self.load_data:
            logger.info('Will NOT load data, as load_data is False.')
            
        else:
            # Create dataset paths
            self.raw_path = os.path.join(self.data_dir, self.raw_name)
            self.train_path = os.path.join(self.data_dir, self.train_name + self.processed_extension)
            self.val_path = os.path.join(self.data_dir, self.val_name + self.processed_extension)
            self.test_path = os.path.join(self.data_dir, self.test_name + self.processed_extension)

            # Load datasets
            if os.path.exists(self.train_path)\
                    and os.path.exists(self.val_path)\
                    and os.path.exists(self.test_path)\
                    and not self.load_raw:
                # Load previously processed sets
                logger.info('Saved preprocessed datasets were found.')
                logger.info('Loading PROCESSED datasets...')
                
                self.loaded_preprocessed = True
                logger.info('loaded_preprocessed = True')
                # Load processed datasets
                self.load_datasets()

            else:
                # 1 - Load raw dataset (must be csv)
                logger.info('No preprocessed data was found.')
                logger.info(f'Reading RAW dataset: {self.raw_path}')
                try:
                    self.data = pd.read_csv(self.raw_path, **self.read_params)
                    logger.info(f'Number of texts: {self.data.shape[0]}')
                except FileNotFoundError:
                    logger.error(f'ERROR: Raw dataset {self.raw_path} not found.')
                    return
                    
                logger.info(f'RAW data loaded.')

                if self.shuffle:
                    logger.info(f'Shuffle...')
                    self.data = self.data.sample(
                        frac=1,random_state = 42).reset_index(drop=True)
                else:
                    logger.info(f'Will NOT Shuffle.')

                #logger.info(f'Determine set example counts.')
    
    @utils.catch('BASEPREPROCESSOR_GETDATASETSERROR')
    def get_datasets(self):
        return dict(
            train_data = self.train_data,
            val_data = self.val_data,
            test_data = self.test_data
        )
    
    @utils.catch('BASEPREPROCESSOR_SPLITDATAERROR')
    def split_data(self):
        """Split into train, val, test. Deletes old data to free memory.
        """
        if not self.load_data:
            raise AttributeError('Preprocessor has not loaded any data.')
        
        # 3 - Find example counts for each set
        self.n_examples = self.data[0].shape[0]
        self.n_train = int(self.n_examples * self.train_ratio)
        self.n_val = int(self.n_examples * self.val_ratio)
        self.n_test = self.n_examples - self.n_train - self.n_val
        
        logger.info(f'Set sizes:')
        logger.info(f'train: {self.n_train}')
        logger.info(f'val: {self.n_val}')
        logger.info(f'test: {self.n_test}')
        if self.n_test < 0:
            raise ValueError('Train + validation ratios must bef < 1')

        # 4 - Separate data into train, test, val
        if isinstance(self.data[0], pd.DataFrame):
            logger.info('Dataset is in a dataframe.')
            self.isdataframe = True

            self.train_data = [self.data[0].iloc[:self.n_train],
                               self.data[1].iloc[:self.n_train]]
            
            self.val_data = [self.data[0].iloc[self.n_train:self.n_val + self.n_train],
                             self.data[1].iloc[self.n_train:self.n_val + self.n_train]]
            
            self.test_data = [self.data[0].iloc[self.n_val + self.n_train:],
                              self.data[1].iloc[self.n_val + self.n_train:]]
            logger.info('Data was split into train, val, test.')
        else:
            self.isdataframe = False
            logger.info('Dataset is in a numpy array.')
            
            # If datasets are numpy array or sparse
            self.train_data = [self.data[0][:self.n_train],
                               self.data[1][:self.n_train]]
            
            self.val_data = [self.data[0][self.n_train:self.n_val + self.n_train],
                             self.data[1][self.n_train:self.n_val + self.n_train]]
            
            self.test_data = [self.data[0][self.n_val + self.n_train:],
                              self.data[1][self.n_val + self.n_train:]]
            logger.info('Data was split into train, val, test.')
            
        assert(self.n_train == self.train_data[0].shape[0])
        assert(self.n_val == self.val_data[0].shape[0])
        assert(self.n_test == self.test_data[0].shape[0])
        
        # Free memory
        del self.data
        
        if self.save_sets:
            self.save_datasets()
    
    @utils.catch('BASEPREPROCESSOR_SAVEDATASETSERROR')
    def save_datasets(self):
        """If datasets are DataFrames, y must be in column 'TARGET'.
        If datasets are sparse, datasets must be (x, y).
        
        Concatenates X and y and saves in a single file for each set.
        """
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
    
    @utils.catch('BASEPREPROCESSOR_LOADDATASETSERROR')
    def load_datasets(self):
        """If datasets are DataFrames, y must be in column 'TARGET'.
        If datasets are sparse, datasets must be (x, y).
        """
        if self.processed_extension == '.npz':
            logger.info(f'Loading sets from npz:')
            
            logger.info(f'train: {self.train_path}')
            self.train_data = sparse.load_npz(self.train_path)

            logger.info(f'val: {self.val_path}')
            self.val_data = sparse.load_npz(self.val_path)

            logger.info(f'test: {self.test_path}')
            self.test_data = sparse.load_npz(self.test_path)
            
            # Split x and y
            self.train_data = [sparse.lil_matrix(sparse.csr_matrix(self.train_data)[:,:-1]),
                               sparse.lil_matrix(sparse.csr_matrix(self.train_data)[:,-1])]
            
            self.val_data = [sparse.lil_matrix(sparse.csr_matrix(self.val_data)[:,:-1]),
                             sparse.lil_matrix(sparse.csr_matrix(self.val_data)[:,-1])]
            
            self.test_data = [sparse.lil_matrix(sparse.csr_matrix(self.test_data)[:,:-1]),
                              sparse.lil_matrix(sparse.csr_matrix(self.test_data)[:,-1])]
            
        elif self.processed_extension == '.csv':
            logger.info(f'Loading sets from csv:')
            
            logger.info(f'train: {self.train_path}')
            self.train_data = pd.read_csv(self.train_path)
            train_cols = self.train_data.columns
            self.train_data = [self.train_data[train_cols.difference(['TARGET'])],
                               self.train_data['TARGET']]
            
            logger.info(f'val: {self.val_path}')
            self.val_data = pd.read_csv(self.val_path)
            self.val_data = [self.val_data[train_cols.difference(['TARGET'])],
                               self.val_data['TARGET']]
            
            logger.info(f'test: {self.test_path}')
            self.test_data = pd.read_csv(self.test_path)
            self.test_data = [self.test_data[train_cols.difference(['TARGET'])],
                               self.test_data['TARGET']]
        else:
            raise AttributeError(f'Wrong extension: {self.processed_extension}')
        self.n_train = self.train_data[0].shape[0]
        self.n_val = self.val_data[0].shape[0]
        self.n_test = self.test_data[0].shape[0]
        self.input_size = self.train_data[0].shape[1]
        self.n_examples = self.n_train + self.n_val + self.n_test
        
        logger.info(f'Set sizes:')
        logger.info(f'train: {self.n_train}')
        logger.info(f'val: {self.n_val}')
        logger.info(f'test: {self.n_test}')
    
    # You must override this method (For prod)
    @utils.catch('BASEPREPROCESSOR_PROCESSERROR')
    def process(self, debug = False, **kwargs):
        raise NotImplementedError()
        
    @utils.catch('BASEPREPROCESSOR_SELECTFEATURESERROR')
    def select_features(self, limit = 100):
        raise NotImplementedError()