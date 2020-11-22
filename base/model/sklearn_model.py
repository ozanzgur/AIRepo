to_import = [
    'utils',
    'model_defaults',
    'base.model.search.bayesian'
    ]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import numpy as np

import logging
logger = logging.getLogger('pipeline')

from sklearn import metrics
import joblib
import pickle
from hyperopt import space_eval

class SklearnModel1:
    """Metric is accuracy, validation is separate. Uses predict_proba.
    
    Suitable models:
    
        - svm:
            from sklearn import svm
            model = svm.SVC(kernel='linear', probability=True, tol=1e-3)
            
        - logistic regression: 
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0,
                tol=0.0001, C=0.12345, max_iter = 100)
                
        - multinomial naive bayes:
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB(alpha=1.0)
                
    """
    @utils.catch('SKELARNMODEL_INITERROR')
    def __init__(self, **hparams):

        hparams_copy = hparams.copy()
        hparams_copy.pop('input_size', None)
        hparams_copy.pop('fig_save_dir', '')
        self.model_tag = hparams_copy.pop('model_tag', 'lr')
        self.minimize_metric = hparams_copy.pop('minimize_metric', False)
        self.model = model_defaults.model_defaults[self.model_tag]['sklearn_model'](**hparams_copy)

    @utils.catch('SKELARNMODEL_FITERROR')
    def fit(self, x, **fit_hparams):
        logger.info(f'Fitting sklearn model...')
        
        X_train, y_train = x['train_data']
        X_val, y_val = x['val_data']
        
        try:
            y_train = np.array(y_train.todense()).ravel()
            y_val = np.array(y_val.todense()).ravel()
        except:
            y_train = np.array(y_train).ravel()
            y_val = np.array(y_val).ravel()
        
        try:
            self.model.fit(X_train, y_train, **fit_hparams)
            
        except ZeroDivisionError: # *** SVM BUG ***
            logger.info(f"SVM error, returning 0.")
            return {'metric': 0.0}
        
        metric = self.model.score(X_val, y_val)
        
        logger.info(f"Metric: {metric}")
        return {'metric': metric}
    
    @utils.catch('SKELARNMODEL_TESTERROR')
    def test(self, x):
        test_data = x['test_data']

        test_preds = self.model.predict(test_data[0])
        return self.model.score(test_data[1], test_preds)

    @utils.catch('SKELARNMODEL_PREDICTERROR')
    def predict(self, x, debug = False, **kwargs):
        """if debug:
            logger.info(f'Model input: {x}')

        for x_example in x:
            x_example.update({'output': self.model.predict(x_example['output'])})"""
        try:
            return self.model.predict_proba(x)
            logger.error('Could not run predict_proba.')
        except:
            return self.model.predict(x)

    @utils.catch('SKELARNMODEL_LOADERROR')
    def load(self, model_path = 'pipeline_model.joblib'):
        # Load model
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            logger.error(f'Model file {model_path} does not exist.')

    @utils.catch('SKELARNMODEL_SAVEERROR')
    def save(self, model_path = 'pipeline_model.joblib'):
        try:
            joblib.dump(self.model, model_path)
        except Exception as e:
            logger.error(e)

    @utils.catch('SKELARNMODEL_FITBESTERROR')
    def fit_best(self, x, trials_path = f'trials_mlp'):

        # Get search space
        search_space = model_defaults.model_defaults[self.model_tag]['search_space']

        # Load trials
        trials = pickle.load(open(trials_path, "rb"))

        # Get best search hparams
        search_params = space_eval(search_space, {k: v[0] for k,v in trials.best_trial['misc']['vals'].items()})
        logger.info(f'Best search hparameters: {search_params}')

        # Join fixed parameters and search parameters
        params = model_defaults.model_defaults[self.model_tag]['search_fixed']
        params.update(search_params)

        self.__init__(
            model_tag = self.model_tag,
            **params
            )
        return self.fit(x)

    @utils.catch('SKELARNMODEL_SEARCHERROR')
    def search(self, x, num_iter = 25, trials_path = 'trials_sklearn', fig_save_dir = ''):        
        # Get default hparams
        search_space = model_defaults.model_defaults[self.model_tag]['search_space']
        fixed_params = model_defaults.model_defaults[self.model_tag]['search_fixed']
        
        # Search
        res_search = bayesian.bayesian_search(
            self.__init__,
            self.fit,
            x,
            search_space,
            fixed_params,
            num_iter = num_iter,
            mode = 'bayesian',
            minimize = self.minimize_metric,
            trials_path = trials_path,
            model_tag = self.model_tag,
            fig_save_dir = fig_save_dir)
        
        return res_search