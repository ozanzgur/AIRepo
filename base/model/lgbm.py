to_import = ['utils',
             'model_defaults',
             'base.model.search.bayesian']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

from lightgbm import LGBMClassifier, LGBMRegressor
import scipy
import pandas as pd
import numpy as np
import shap

shap.initjs()

import logging
logger = logging.getLogger('pipeline')
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class LGBM:
    @utils.catch('LGBM_INITERROR')
    def __init__(self, **kwargs):
        def_params = dict(
            models = [],
            #folds = KFold(n_splits=2, random_state=42, shuffle=True),
            #objective = 'binary',
            #metric = 'accuracy',
            #binary_threshold = 0.5,
            features = None,
            metric_func = None,
            minimize_metric = False
        )
        
        lgbm_hparam_keys = [
            'num_leaves',
            'max_depth',
            'min_data_in_leaf',
            'bagging_fraction',
            'learning_rate',
            'reg_alpha',
            'reg_lambda',
            'min_sum_hessian_in_leaf',
            'feature_fraction',
            'unbalanced_sets',
            'num_iterations',
            'random_state',
            'bagging_freq',
            'bagging_seed',
            'early_stopping_round',
            'objective',
            'metric',
            'verbose',
            'num_class'
        ]
        
        lgbm_hparams = {}
        for k in lgbm_hparam_keys:
            if k in kwargs:
                lgbm_hparams[k] = kwargs[k]
        self.lgbm_hparams = lgbm_hparams
        
        self.__dict__.update(def_params)
        self.__dict__.update(kwargs)
        
        assert(self.objective in ['binary', 'multiclass', 'regression', 'multiclassova'])
        self.feature_importances = None
        
    @utils.catch('LGBM_TESTERROR')
    def test(self, test_data):
        X_test = test_data[0]
        y_test = test_data[1]
        
        test_metric = None
        test_preds = self.predict(X_test)
        return self.get_metric(test_preds, y_test)
    
    @utils.catch('LGBM_METRICERROR')
    def get_metric(self, y_true, y_pred):
        if self.metric_func is None:
            if self.metric == 'accuracy':
                if self.objective in ['multiclass', 'multiclassova']:
                    return accuracy_score(y_true, np.argmax(y_pred, axis = 1))
                elif self.objecetive == 'binary':
                    return accuracy_score(y_true, y_pred > self.binary_threshold)
                
        else:
            return self.metric_func(y_true, y_pred)
    
    @utils.catch('LGBM_PREDICTERROR')
    def predict(self, x, debug = False, **kwargs):
        model_preds = []
        assert(len(self.models) > 0)
        logger.info(f'Model count: {len(self.models)}')
        
        # Predict for each model
        model_preds = []
        for model in self.models:
            try:
                model_preds.append(model.predict_proba(x))
            except:
                model_preds.append(model.predict(x))

        preds = np.mean(model_preds, axis = 0)
        
        # Return results of each prediction
        return preds
    
    @utils.catch('LGBM_FITERROR')
    def fit(self, x):
        """ X can be pd.Dataframe, np.ndarray or sparse.
        y has to be pd.series
        """
        train_data = x['train_data']
        val_data = x['val_data']
        
        self.models = []
        
        # For CV
        oof_preds = np.zeros(len(train_data[0]))
        X_data = train_data[0]
        y_data = train_data[1]
        
        # Validate after CV
        X_val = val_data[0]
        try:
            y_val = np.array(val_data[1].todense()).ravel()
        except:
            y_val = np.array(val_data[1]).ravel()
        
        is_sparse = scipy.sparse.issparse(X_data)
        
        # Create dataframe to keep feature importances for each fold
        feature_importances = pd.DataFrame()
        if not is_sparse:
            self.features = X_data.columns
            
        if self.features is not None:
            if not len(self.features) == X_data.shape[1]:
                raise ValueError(
                    'Number of features must be the same as n_columns in X.')
        
            # Create column for features
            feature_importances['feature'] = self.features
            
        cv_metrics = list()

        n_folds = 0
        folds = None
        val_preds = None
        
        if not isinstance(self.folds, list):
            folds = self.folds.split(X_data, y_data)
        else:
            folds = self.folds
            
        for i_fold, (trn_idx, val_idx) in enumerate(folds):
            n_folds += 1
            X_trn_fold = X_data[trn_idx] if is_sparse else X_data.iloc[trn_idx]
            X_val_fold = X_data[val_idx] if is_sparse else X_data.iloc[val_idx]
            
            y_val_fold = None
            y_trn_fold = None
            if isinstance(y_data, pd.Series):
                y_trn_fold = y_data.iloc[trn_idx]
                y_val_fold = y_data.iloc[val_idx]
            else:
                y_trn_fold = y_data[trn_idx]
                y_val_fold = y_data[val_idx]
                try:
                    y_trn_fold = np.array(y_trn_fold.todense()).ravel()
                    y_val_fold = np.array(y_val_fold.todense()).ravel()
                except:
                    y_trn_fold = np.array(y_trn_fold).ravel()
                    y_val_fold = np.array(y_val_fold).ravel()
            
            logger.info('Training on fold {}'.format(i_fold))
            """trn_data = lgb.Dataset(X_trn_fold, label = y_trn_fold)
            val_data = lgb.Dataset(X_val_fold, label = y_val_fold)"""
            # This is validation in CV, not validation set
            
            # Training for this fold
            #print(self.lgbm_hparams)
            clf = LGBMRegressor(**self.lgbm_hparams) if self.objective == 'regression' else LGBMClassifier(**self.lgbm_hparams)
            clf = clf.fit(X = X_trn_fold, y = y_trn_fold,
                          eval_set  = [(X_trn_fold, y_trn_fold),
                                       (X_val_fold, y_val_fold)],
                          early_stopping_rounds = 250,
                          verbose = 200)
            
            # Keep models of each fold
            self.models.append(clf)

            feature_importances['fold_{}'.format(i_fold)] = clf.feature_importances_
            
            try:
                oof_preds[val_idx] = clf.predict_proba(X_val_fold)
            except:
                oof_preds[val_idx] = clf.predict(X_val_fold)
            
            # Validation for this fold
            if X_val is not None:
                if val_preds is None:
                    try:
                        val_preds = clf.predict_proba(X_val)
                    except:
                        val_preds = clf.predict(X_val)
                else:
                    try:
                        val_preds += clf.predict_proba(X_val)
                    except:
                        val_preds += clf.predict(X_val)
                    
        logger.info('Training has finished.')
        #logger.info(f'Mean CV {params["metric"]}: {np.mean(cv_metrics)}')
        
        # Validation
        val_metric = None
        if X_val is not None:
            val_preds /= n_folds
            
            logger.info('Calculating validation metric...')
            val_metric = self.get_metric(y_val, val_preds)
            
            logger.info(f'Validation {self.metric}: {val_metric}')
            
        feature_importances['importance'] = \
            feature_importances[[f'fold_{i}' for i in range(n_folds)]].sum(axis = 1)
        
        cols_to_keep = [col for col in feature_importances.columns if 'fold' not in col]
        self.feature_importances = feature_importances[cols_to_keep]
        
        if 'feature' in self.feature_importances.columns:
            self.feature_importances.sort_values(by = 'importance',
                                                 ascending = False,
                                                 inplace = True)
        return {
            #'cv_metrics': cv_metrics,
            'feature_importances': feature_importances,
            'val_preds' : val_preds,
            'oof_preds': oof_preds,
            'metric': val_metric
        }
    
    @utils.catch('LGBM_DISPLAYFEATUREIMPORTANCESERROR')
    def display_feature_importances(self):
        display(self.feature_importances.style.background_gradient(cmap = 'coolwarm'))
        
    
    @utils.catch('LGBM_EXPLAINSHAPERROR')
    def explain_shap(self, data, features = None, class_names = None, which_class = None):
        X, y = data
        
        explainer = shap.TreeExplainer(self.models[0])
        shap_values = explainer.shap_values(X)
        if which_class is not None:
            assert(class_names is not None)
            assert(which_class in class_names)
            class_i = class_names.index(which_class)
            shap_values = shap_values[class_i]
            
        shap.summary_plot(shap_values,
                          X,
                          feature_names = features,
                          class_names = class_names)
    
    @utils.catch('LGBM_SEARCHERROR')
    def search(self, x, num_iter = 3, trials_path = 'trials_lgbm', fixed_hparams = None):        
        # Get default hparams
        search_space = model_defaults.model_defaults['lgbm']['search_space']
        fixed_params = model_defaults.model_defaults['lgbm']['search_fixed'] if fixed_hparams is None else fixed_hparams
        
        print("Fixed hparameters:")
        print(fixed_params)
        self.__dict__.update(fixed_params)
        
        # Search
        print('Minimize metric:')
        print(self.minimize_metric)
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
            model_tag = 'lgbm')
        
        return res_search