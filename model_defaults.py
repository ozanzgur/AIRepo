to_import = ['utils',
             'base.model.mlp',
             'base.model.sklearn_model',
             'base.model.lgbm']

from os.path import dirname, abspath
project_name = dirname(abspath(__file__)).split('\\')[-1]

import sys
import importlib
for module in to_import:
    module_name = module.split('.')[-1]
    new_module = importlib.import_module(name = f'.{module}', package = project_name)
    sys.modules[__name__].__dict__.update({module_name: new_module})

################################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier


from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.metrics import accuracy_score

output_size = 4
score = accuracy_score

model_defaults = {
    # Fixed parameters are not changed during hparam search
    # Search is carried out in search_space
    
    # LOGISTIC REGRESSION ######################################################
    'lr': {
        'model': sklearn_model.SklearnModel1,
        'sklearn_model': LogisticRegression,
        'hparams': dict(
            penalty = 'l2',
            tol = 0.0001,
            C = 0.12345,
            class_weight = None, # 'balanced'
            random_state = 42,
            solver = 'lbfgs',
            max_iter = 500,
            multi_class = 'auto',
            verbose = 0,
            n_jobs = -1
        ),
        'search_space': dict(
            C = hp.loguniform('C', -7, 3),
            class_weight =  hp.choice('class_weight', ['balanced', None]),
            solver =  hp.choice('solver', ['newton-cg', 'lbfgs', 'sag']),
            tol = hp.loguniform('tol', -7, 1)
        ),
        'search_fixed': dict(     
            max_iter = 500,
            verbose = 0,
            n_jobs = -1,
            penalty = 'l2',
            multi_class = 'auto',
            random_state=42,
        )
    },
    
    # SVM ######################################################################
    'svm': {
        'model': sklearn_model.SklearnModel1,
        'sklearn_model': SVC,
        'hparams': dict(
            kernel='linear',
            tol=1e-3,
            C = 1.0,
            gamma = 'scale',
            random_state = 42,
            class_weight = 'balanced',
            probability = True
        ),
        'search_space': dict(
            kernel =  hp.choice('kernel ', ['linear', 'rbf', 'poly']),
            class_weight =  hp.choice('class_weight', ['balanced', None]),
            C = hp.loguniform('C', -7, 3),
            tol = hp.loguniform('tol', -7, 3),
            
        ),
        'search_fixed': dict(
            gamma = 'scale',
            random_state = 42,
            probability = True
        )
    },
    
    # NB #######################################################################
    
    'nb': {
        'model': sklearn_model.SklearnModel1,
        'sklearn_model': MultinomialNB,
        'hparams': dict(
            alpha = 1.0
        ),
        'search_space': dict(
            alpha = hp.loguniform('alpha', -7, 3)
        ),
        'search_fixed': dict(
            
        )
    },
    
    # LGBM #####################################################################
    'lgbm': {
        'model': lgbm.LGBM,
        #'fit_hparams': ['early_stopping_rounds', 'verbose'],
        'hparams': dict(
            num_leaves = 10, # num_leaves must be < 2**max_depth
            max_depth = 5,
            min_data_in_leaf = 200, 
            bagging_fraction = 0.9,
            learning_rate = 1e-2,
            reg_alpha = 0,
            reg_lambda = 0,
            num_iterations = 10000,
            random_state = 42,
            min_sum_hessian_in_leaf = 1e-3,
            bagging_freq = 5,
            bagging_seed = 42,
            feature_fraction = 1.0, # Training too slow? reduce this
            #early_stopping_round = 500,
            #verbose = 0,
            unbalanced_sets = True,
            metric = 'multi_error',
            objective = 'multiclassova', # multiclass,
            num_class = output_size
            
        ),
        'search_space': dict(
            num_leaves = scope.int(hp.quniform('num_leaves', 2, 100, 1)),
            max_depth = scope.int(hp.quniform('max_depth', 2, 20, 1)),
            min_data_in_leaf = scope.int(hp.quniform('min_data_in_leaf', 3, 1000, 1)),
            bagging_fraction = hp.uniform('bagging_fraction', 0.025, 1.0),
            learning_rate = hp.loguniform('learning_rate', -4, 1),
            reg_alpha = hp.loguniform('reg_alpha', -7, 3),
            reg_lambda = hp.loguniform('reg_lambda', -7, 3),
            min_sum_hessian_in_leaf = hp.loguniform('min_sum_hessian_in_leaf', -5, 1),
            feature_fraction = hp.uniform('feature_fraction', 0.001, 1.0),
            unbalanced_sets = hp.choice('unbalanced_sets ', [True, False]),
        ),
        'search_fixed': dict(
            num_iterations = 10000,
            random_state = 42,
            bagging_freq = 5,
            bagging_seed = 42,
            #early_stopping_round = 500,
            metric = 'multi_error',
            objective = 'multiclassova',
            #verbose = 0,
            num_class = output_size
        )
    },
    
    # MLP ######################################################################
    'mlp': {
        'model': mlp.MLP,
        'hparams': dict(
            n_layers = 1,
            size = 10,
            shrink_rate = 0.7,
            learning_rate = 1e-3,
            beta_1 = 0.95,
            beta_2 = 0.999,
            batch_size = 25,
            dropout = 0.25,
            metric = 'categorical_accuracy',
            loss = 'categorical_crossentropy',
            activation = 'relu',
            output_size = output_size,
            metric_minimize = False,
            verbose = 2, # 2 => one line per epoch, 1 => progress bar, 0 => silent
            use_callback_checkpoint = False,
            ohe = True
        ),
        'search_space': dict(
            # Architecture
            n_layers = scope.int(hp.quniform('n_layers', 1, 3, 1)),
            size = scope.int(hp.quniform('size', 3, 125, 1)),
            shrink_rate = hp.uniform('shrink_rate', 0.05, 0.999),
            activation = hp.choice('activation ', ['relu', 'tanh', 'sigmoid']),
            dropout = hp.uniform('dropout', 0.001, 0.96),
            
            # Training
            learning_rate = hp.loguniform('learning_rate', -4, 1),
            beta_1 = hp.uniform('beta_1', 0.85, 0.99),
            beta_2 = hp.uniform('beta_2', 0.992, 0.9999),
            batch_size = scope.int(hp.quniform('batch_size', 5, 100, 1))
        ),
        'search_fixed': dict(
            #output_size = output_size,
            metric = 'categorical_accuracy',
            loss = 'categorical_crossentropy',
            minimize_metric = False,
            verbose = 2,
            use_callback_checkpoint = False,
            ohe = True
        )
    },
    # CRF ######################################################
    'crf': {
        'model': sklearn_model.SklearnModel1,
        'sklearn_model': CRF,
        'hparams': dict(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        ),
        'search_space': hp.choice('type',
                [{
                    'algorithm': 'lbfgs',
                    'delta': hp.loguniform('delta_1', -7, -3),
                    'epsilon':  hp.loguniform('epsilon_1', -7, -3),
                    'c2': hp.loguniform('c2_1', -7, 3),
                    'all_possible_transitions': hp.choice('all_possible_transitions_1', [True, False]),
                    'all_possible_states': hp.choice('all_possible_states_1', [True, False]),
                    'min_freq': scope.int(hp.quniform('min_freq_1', 0, 4, 1)),
                    'c1': hp.loguniform('c1', -7, 3),
                    'num_memories': scope.int(hp.quniform('num_memories', 3, 9, 1)),
                    'period': scope.int(hp.quniform('period', 5, 15, 1)),
                    'linesearch': hp.choice('linesearch', ['MoreThuente', 'Backtracking', 'StrongBacktracking']),
                    'max_linesearch': scope.int(hp.quniform('max_linesearch', 10, 30, 1)),
                 },
                {
                    'algorithm': 'l2sgd',
                    'c2': hp.loguniform('c2_2', -7, 3),
                    'delta': hp.loguniform('delta_2', -7, -3),
                    'all_possible_transitions': hp.choice('all_possible_transitions_2', [True, False]),
                    'all_possible_states': hp.choice('all_possible_states_2', [True, False]),
                    'min_freq': scope.int(hp.quniform('min_freq_2', 0, 4, 1)),
                    'calibration_eta': hp.loguniform('calibration_eta', -2, 0),
                    'calibration_rate': hp.loguniform('calibration_rate', -1, 1),
                    'calibration_samples': scope.int(hp.quniform('calibration_samples', 300, 2000, 1)),
                    'calibration_candidates': scope.int(hp.quniform('calibration_candidates', 5, 15, 1)),
                    'calibration_max_trials': scope.int(hp.quniform('calibration_max_trials', 10, 30, 1)),
                 },
                 {
                     'algorithm': 'pa',
                     'epsilon':  hp.loguniform('epsilon_2', -7, -3),
                     'all_possible_transitions': hp.choice('all_possible_transitions_3', [True, False]),
                     'all_possible_states': hp.choice('all_possible_states_3', [True, False]),
                     'min_freq': scope.int(hp.quniform('min_freq_3', 0, 4, 1)),
                     'pa_type': hp.choice('pa_type', [0, 1, 2]),
                     
                     'c': hp.loguniform('c', -1, 1),
                     'error_sensitive': hp.choice('error_sensitive', [True, False]),
                     'averaging': hp.choice('averaging', [True, False]),
                 },
                 {
                     'algorithm': 'arow',
                     'epsilon':  hp.loguniform('epsilon_3', -7, -3),
                     'all_possible_transitions': hp.choice('all_possible_transitions_4', [True, False]),
                     'all_possible_states': hp.choice('all_possible_states_4', [True, False]),
                     'min_freq': scope.int(hp.quniform('min_freq_4', 0, 4, 1)),
                     'variance': hp.loguniform('variance', -0.5, 0.5),
                     'gamma': hp.loguniform('gamma', -2, 2),
                 },
                 {
                     'algorithm': 'ap',
                     'epsilon':  hp.loguniform('epsilon_4', -7, -3),
                     'all_possible_transitions': hp.choice('all_possible_transitions_5', [True, False]),
                     'all_possible_states': hp.choice('all_possible_states_5', [True, False]),
                     'min_freq': scope.int(hp.quniform('min_freq_5', 0, 4, 1)),
                 }]
        ),
        'search_fixed': dict(
        )
    },
}