from hyperopt import tpe, fmin
from hyperopt import Trials
from hyperopt import space_eval
import copy
import numpy as np
import pickle

import logging
logger = logging.getLogger('pipeline')
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

def extract_best_params(trials, search_space):
    best_vals = {k: v[0] for k,v in trials.best_trial['misc']['vals'].items()}
    return space_eval(search_space, best_vals)

def extract_best_result(trials):
    return trials.best_trial['result']['loss']

def bayesian_search(
        init_function, fit_function, data, search_params, fixed_params,
        num_iter = 25, mode = 'bayesian', minimize = True, trials_path = 'trials', **kwargs):
    
    print('Minimize metric:')
    print(minimize)
    
    space = search_params
    results = dict.fromkeys(['best_params', 'best_metric'])
    all_metrics = []
    
    # Load previos state of search
    tpe_trials = None
    trials_loaded = False
    try:
        tpe_trials = pickle.load(open(trials_path, "rb"))
        all_metrics = [r['loss'] for r in tpe_trials.results if 'loss' in r]
        
        # Get best result and parameters from loaded trials
        print('Getting search state...')
        results['best_params'] = extract_best_params(tpe_trials, space)
        results['best_metric'] = extract_best_result(tpe_trials)
        trials_loaded = True
        
        print('Best hparameters:')
        print(results['best_params'])
        print('Best metric:')
        print(results['best_metric'])
        
        if not minimize:
            all_metrics = [-m for m in all_metrics]
        
        logger.info(f'SEARCH: LOADED TRIALS FILE FROM {trials_path}.')
    except FileNotFoundError:
        logger.info('SEARCH: Trials file does not exist.')
    
    if tpe_trials is None:
        tpe_trials = Trials()
    else:
        logger.info(f'Continue search from iteration: {len(tpe_trials.results)}')
    
    
    # mode must be one of ['bayesian', 'random']
    if not (mode in ['bayesian', 'random']): raise AssertionError()
    
    first_run = True

    # Plot search results
    x = np.arange(len(all_metrics))
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dots1, = ax.plot(x, all_metrics, 'bo')

    ############################
    
    # Hyperopt optimizes the result of this function
    def objective(params):
        nonlocal first_run
        
        # Print info if it exists
        if trials_loaded or not first_run:
            first_run = False
            
            print('Best hparams:')
            print(extract_best_params(tpe_trials))
            print('Best result:')
            print(extract_best_result(tpe_trials))
        
        # Add fixed parameters to search parameters
        params_comb = params.copy()
        params_comb.update(fixed_params)
        
        # Initialize model and fit
        init_function(**params_comb, **kwargs)
        
        # Fit model
        param_metric = fit_function(data)['metric']
        all_metrics.append(param_metric)
        
        # Decide if result is the best one
        improve = False
        if results['best_metric'] is None:
            improve = True
        elif minimize and (param_metric < results['best_metric']):
            improve = True
        elif not minimize and (param_metric > results['best_metric']):
            improve = True
            
        if improve:
            results['best_metric'] = param_metric
            results['best_params'] = params_comb
        
        # Display history
        clear_output()
        print(f'Minimize: {minimize}')
        dots1.set_ydata(all_metrics)
        dots1.set_xdata(np.arange(len(all_metrics)))
        ax.set_xlim(-0.5, len(all_metrics) - 0.5)
        ax.set_title('Hyperparameter Search')
        ax.set_xlabel('Trial index')
        ax.set_ylabel('Metric')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.5)
        plt.show(block=False) # block=False

        """plt.scatter(y = all_metrics, x = np.arange(len(all_metrics)))
        plt.show()
        plt.pause(0.0001)"""
        
        # Save trials
        try:
            pickle.dump(tpe_trials, open(trials_path, "wb"))
            logger.info(f'Saved trials to {trials_path}')
        except Exception as e:
            logger.error(f'Trials could not be saved to {trials_path}')
            logger.error(e)
        
        if minimize:
            return param_metric
        else:
            return - param_metric
    
    # Start hparam search
    tpe_algo = tpe.suggest if mode == 'bayesian' else tpe.random.suggest
    fmin(fn=objective, space=space, 
         algo=tpe_algo, trials=tpe_trials,
         max_evals=num_iter)
    
    logger.info(f'Best metric: {results["best_metric"]}')
    logger.info('<b>SELECTED HPARAMETERS: </b>')
    logger.info(results['best_params'])
    
    results['tpe_trials'] = tpe_trials
    return results