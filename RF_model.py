import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import shap
from scipy.stats import linregress, pearsonr


np.random.seed(0)


def evaluate(model, features, labels, model_name = 'Model'):
    predictions = model.predict(features)
    errors = abs(predictions - labels)
    mape = 100 * np.mean(errors / labels)
    accuracy = 100 - mape
    print('%s Performance' % model_name)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy, np.mean(errors)


def get_gof_metrics(pred, obs, prefix=None):
    n = len(obs)
    bias = np.mean(pred - obs)
    mae = np.mean(abs(pred - obs))
    rmse = np.sqrt(((pred - obs) ** 2).mean())
    mape = np.mean(abs(pred - obs) / obs) * 100
    accuracy = 100 - mape
    nse = r2_score(obs, pred)
    r = np.corrcoef(pred, obs)[0,1]
    r2 = r**2
    slope,_,_,_,_ = linregress(obs, pred)
    pearsonrho = pearsonr(obs, pred)
    mean = np.mean(obs)
    nRmse = rmse/mean
    precision = np.sqrt(np.sum((pred-obs-bias) ** 2)/n)
    
    values = [n, rmse, bias, precision, mae, mape, nRmse, accuracy, r2, r, pearsonrho, nse, slope]
    names = ['count', 'rmse', 'bias', 'precision', 
             'mae', 'mape', 'nRmse', 'accuracy',
             'r2', 'rho', 'persrho','nse', 
             'slope']
    if prefix != None:
        names = [prefix+'_'+x for x in names]
    
    out = pd.Series(dict(zip(names, values)))    
    
    return out


def tune_hp_grid(train_features, train_labels, model_name, cv=3, n_jobs=-1, verbose=0, print_imp=None):
    param_grid = {'n_estimators': [3, 4, 5, 6, 7, 8, 9, 10], 
     'max_features': ['auto', 'sqrt'], 
     'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 
     'min_samples_split': [10, 15, 20, 25, 30], 
     'min_samples_leaf': [5, 10, 15, 20, 25, 30, 35, 40], 
     'bootstrap': [True, False]}

    rf = RandomForestRegressor()

    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                               cv = cv, n_jobs = n_jobs, verbose = verbose)


    grid_search.fit(train_features, train_labels)
    params_rf = grid_search.best_params_
    
    np.save('../outputs/Results/tuned_params/tuned_parameters_%s.npy' % model_name, params_rf)
    
    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(train_features, train_labels)
    base_accuracy, base_errors = evaluate(base_model, train_features, train_labels, model_name = 'Base Model')

    best_grid = grid_search.best_estimator_
    grid_accuracy, grid_errors = evaluate(best_grid, train_features, train_labels, model_name = '%s Model' % model_name)
    if print_imp:
        print('')
        print('Improvement of {:0.2f}%. acurary'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
        print('Improvement of {:0.2f}%. average error'.format( 100 * (grid_errors - base_errors) / base_errors))
    return params_rf


def run_random_forest_i(features, labels, train_index, test_index, hp_params, features_list):
    rf = RandomForestRegressor(n_estimators = hp_params["n_estimators"], 
                                   min_samples_split = hp_params["min_samples_split"], 
                                   min_samples_leaf = hp_params["min_samples_leaf"], 
                                   max_depth = hp_params["max_depth"]) 

    features_train = features[train_index]
    features_test = features[test_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]
        
    rf.fit(features_train, np.ravel(labels_train[:,1:]))
    predictions = {'all': rf.predict(features), 'train': rf.predict(features_train), 'test': rf.predict(features_test)}
    
    importances = list(rf.feature_importances_)  
    feature_importances = [(features, round(importance, 2)) 
                           for features, importance 
                           in zip(features_list, importances)]
    # Sort the feature importances by most important first
    ##feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances  
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    perm = permutation_importance(rf, features_test, predictions['test'],
                                n_repeats=30, random_state=0)  
    
    metrics_all = get_gof_metrics(predictions['all'], np.ravel(labels[:,1:]))
    metrics_train = get_gof_metrics(predictions['train'], np.ravel(labels_train[:,1:]))
    metrics_test = get_gof_metrics(predictions['test'], np.ravel(labels_test[:,1:]))
    metrics = {'all': metrics_all, 'train': metrics_train, 'test': metrics_test}

    return rf, predictions, metrics, feature_importances, perm
        

def run_random_forest_kfold(dataF, features_list, target_name, n_splits=5, n_splits_cv=3, model_variant='m'):

    features = np.array(dataF[features_list])
    labels_ = np.ravel(np.array(dataF[[target_name, ]]))
    labels = np.array(dataF[['year', target_name]])
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    k_results = []
    for ki, (train_index, test_index) in enumerate(kf.split(features)):
        print(ki, train_index)
        hp_params = tune_hp_grid(features[train_index], labels_[train_index], '%s_k%s' %(model_variant, ki), cv=n_splits_cv, n_jobs=-1, verbose=0)

        rf, predictions, metrics, feature_importances, perm = run_random_forest_i(features, labels, train_index, test_index, hp_params, features_list)
        k_results.append([rf, predictions, metrics,feature_importances, perm] )
        dataF['RF_LNC_%s_all'%ki] = predictions['all']
        dataF['RF_LNC_%s_test'%ki] = [p if ip in test_index else 0 for ip, p in enumerate(predictions['all'])]
        dataF['RF_LNC_%s_train'%ki] = [p if ip in train_index else np.nan for ip, p in enumerate(predictions['all'])]

    dataF['RF_LNC%s' % model_variant] = dataF['RF_LNC_0_test'] + dataF['RF_LNC_1_test'] + dataF['RF_LNC_2_test'] + dataF['RF_LNC_3_test'] + dataF['RF_LNC_4_test']
        
    return k_results, dataF
    