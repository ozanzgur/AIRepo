to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import os
import pandas as pd
import logging
from scipy import sparse
logger = logging.getLogger('pipeline')

class CSVLoader:
    def __init__(self, **kwargs):
        def_args = dict(
            target_column = 'target',
            raw_name = 'raw_data.csv',
            data_dir = '',
            train_ratio = 0.75,
            val_ratio = 0.125,
            classes = None,
            shuffle = True,
            train_name = 'train',
            val_name = 'val',
            test_name = 'test',
            processed_extension = '.npz',
            word_count_limit = 5,
            label_mapping = None,
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
        
        self.__dict__.update(def_args)
        self.__dict__.update(kwargs)
        
        # Extension must be .csv or .npz
        assert(self.processed_extension in ['.csv', '.npz'])
        
        # Load raw data. If preprocessed version exists, load that one.
        # Create dataset paths
        self.raw_path = os.path.join(self.data_dir, self.raw_name)
        self.train_path = os.path.join(self.data_dir, self.train_name + self.processed_extension)
        self.val_path = os.path.join(self.data_dir, self.val_name + self.processed_extension)
        self.test_path = os.path.join(self.data_dir, self.test_name + self.processed_extension)
        
    def __call__(self, x = None, debug = False):
        if x is None:
            x = {}
        
        # Load datasets
        if os.path.exists(self.train_path)\
                and os.path.exists(self.val_path)\
                and os.path.exists(self.test_path)\
                and not self.load_raw:
            # Load previously processed sets
            logger.info('Saved preprocessed datasets were found.')
            logger.info('Loading PROCESSED datasets...')
            
            # Load processed datasets
            self._load()
            
            self.loaded_preprocessed = True
            logger.info('loaded_preprocessed = True')

        else:
            # 1 - Load raw dataset (must be csv)
            logger.info('No preprocessed data was found.')
            logger.info(f'Reading RAW dataset: \n{self.raw_path}')
            try:
                self.data = pd.read_csv(self.raw_path, **self.read_params)

                logger.info(f'Raw dataset slice: {str(self.data.iloc[:3])}')
                logger.info(f'Number of examples: {self.data.shape[0]}')

                if self.label_mapping is not None:
                    logger.info(f'Label mapping: \n{self.label_mapping}')
                    logger.info('Mapping labels...')
                    self.data[self.target_column] = self.data[self.target_column].map(self.label_mapping)
                logger.info(f'Unique labels: {self.data[self.target_column].unique()}')
            except FileNotFoundError:
                logger.error(f'ERROR: Raw dataset {self.raw_path} not found.')
                return
                
            logger.info(f'RAW data loaded.')
            
            # Shuffle dataset
            if self.shuffle:
                logger.info(f'Shuffle...')
                self.data = self.data.sample(
                    frac=1,random_state = 42).reset_index(drop=True)
            else:
                logger.info(f'Will NOT Shuffle.')

            logger.info(f'Dataset slice: \n{str(self.data.iloc[:3])}')
        
        # If existing preprocessed dataset was found and loaded
        if self.loaded_preprocessed:
            assert(self.train_data is not None)
            logging.info('LOADED_PREPROCESSED = True')
            x['train_data'] = self.train_data
            x['val_data'] = self.val_data
            x['test_data'] = self.test_data
            x['loaded_preprocessed'] = True

        else:
            assert(self.data is not None)
            x['data'] = self.data
            x['loaded_preprocessed'] = False

        return x
    
    def _load(self):
        """If datasets are DataFrames, y must be in column 'TARGET'.
        If datasets are sparse, datasets must be (x, y).
        """
        # Data is sparse
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
        
        # Data is csv
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

        logger.info(f'Data sizes:')
        logger.info(f'train: {self.train_data[0].shape}')
        logger.info(f'val: {self.val_data[0].shape}')
        logger.info(f'test: {self.test_data[0].shape}')