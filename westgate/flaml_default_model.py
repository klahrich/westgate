import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from flaml import AutoML
from flaml.automl.ml import norm_confusion_matrix
from sklearn.model_selection import train_test_split
from flaml.ml import sklearn_metric_loss_score
import argparse
from typing import Dict, List
from sklearn.metrics import classification_report
import locale
from flaml.automl.data import get_output_from_log
import plotext as plt
from termcolor import colored
pd.set_option('mode.chained_assignment', None)
    


def prepare_data(features_file:str, split_ratio=0.15) -> pd.DataFrame:

    original_df = pd.read_csv('AttributesLoans2022.csv', encoding='latin', 
                                dtype={'loginId': 'str'})

    df = original_df[original_df.error.isna()]

    df = df[~df.Id.isna()]

    features_df = pd.read_csv(features_file)

    thresholds_min_df = features_df.loc[~features_df.threshold_min.isna(), ['variable', 'threshold_min']]
    thresholds_min = thresholds_min_df.to_dict(orient='records')

    thresholds_max_df = features_df.loc[~features_df.threshold_max.isna(), ['variable', 'threshold_max']]
    thresholds_max = thresholds_max_df.to_dict(orient='records')

    for r in thresholds_min:
        df = df[df[r['variable']] >= r['threshold_min']]

    for r in thresholds_max:
        df = df[df[r['variable']] <= r['threshold_max']]

    target = features_df.loc[features_df.type=='target', 'variable'].item()

    extra_features = features_df.loc[features_df.type=='extra', 'variable']

    df = df[[c for c in df.columns if c in list(features_df['variable'])]]

    df['profit'] = df['total_paid'] - df['principal']

    df[target] = df[target].map({'Fully Paid':0, 'Restructured':0, 'Defaulted':1})

    X = df[[c for c in df.columns if c != target]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42, stratify=y)

    extra = {}

    extra['test_profit'] = X_test['profit']

    for f in list(extra_features):
        extra['test_' + f] = X_test[f]

    X_train = X_train[[c for c in X_train.columns if c not in list(extra_features) and c != 'profit']]
    X_test = X_test[[c for c in X_train.columns if c not in list(extra_features) and c != 'profit']]

    extra['features'] = X_train.columns

    assert 'total_paid' not in X_train.columns
    assert 'status' not in X_train.columns
    assert 'profit' not in X_train.columns

    return X_train, X_test, y_train, y_test, extra


def avg_profit_func(y_pred, y_val): 
    if y_pred == 0:
        if y_val == 0:
            return 385
        else:
            return -120
    else:
        return 0


def profit_func(y_pred, y_profit): 
    if y_pred == 0:
        return y_profit
    else:
        return 0


def make_custom_metric(threshold):

    def custom_metric(X_val, y_val, estimator, labels,
                    X_train, y_train, weight_val=None, weight_train=None,
                    *args):

        y_probas = estimator.predict_proba(X_val)[:, 1]

        decision_lambda = np.vectorize(lambda p: 1 if p > threshold else 0)

        y_pred = decision_lambda(y_probas)

        profit_pred = np.vectorize(avg_profit_func)(y_pred, y_val).sum()

        profit_actual = np.vectorize(avg_profit_func)(0, y_val).sum()

        extra = {'profit_val_pred': int(profit_pred), 'profit_val_actual': int(profit_actual)}
        metric = (profit_actual - profit_pred) if not np.all(y_pred == 0) else 99999999

        return metric, extra

    return custom_metric


def plot_learning_curve(log_file_name, time_budget):
    time_history, best_valid_loss_history, _, _, _ = get_output_from_log(filename=log_file_name,
                                                                         time_budget=time_budget)
    plt.title("Learning Curve")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Validation metric")
    plt.plotsize(100, 30)
    plt.plot(time_history, 1 - np.array(best_valid_loss_history))
    plt.show()


def plot_feat_imp(automl):
    #plt.title("Feature Importance")
    #plt.xlabel("Feature")
    #plt.ylabel("Importance")
    plt.plotsize(100, 30)

    feat_imp_raw = automl.model.estimator.feature_importances_
    idx_sorted = np.argsort(feat_imp_raw)[::-1]
    idx_sorted2 = np.argsort(feat_imp_raw)

    feat_imp = feat_imp_raw[idx_sorted][:10]
    feat_names = automl.model.estimator.feature_names_in_[idx_sorted][:10]

    feat_imp2 = feat_imp_raw[idx_sorted2][:10]
    feat_names2 = automl.model.estimator.feature_names_in_[idx_sorted2][:10]

    print(colored('\nFeature imporances:', 'blue'))

    print('\n**Most important**')
    for k,v in zip(feat_names, feat_imp):
        print(k + ':' + str(round(v, 3)))

    print('\n**Least important**')
    for k,v in zip(feat_names2, feat_imp2):
        print(k + ':' + str(round(v, 3)))


def main(
    features_file:str,
    experiment_id:str,
    automl_config:Dict = {},
    threshold = 0.5):

    X_train, X_test, y_train, y_test, extra = prepare_data(features_file)

    print('X_train shape: ' + str(X_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('----------')

    log_file_name = 'perf-' + experiment_id + '-' + features_file + '.log'

    automl_settings = {
        "time_budget": 60,  # in seconds
        "metric": make_custom_metric(threshold),
        "task": 'classification',
        "estimator_list": ['xgboost'],
        "log_file_name": log_file_name,
        "eval_method": "cv",
        "n_splits": 5,
        "retrain_full": True,
        "log_type": "all",
        "verbose": 2,
        "seed": 123,
        "starting_points": {
            "xgboost": {
                "n_estimators": 4, 
                "max_leaves": 125, 
                "min_child_weight": 0.001, 
                "learning_rate": 1.0, 
                "subsample": 1.0, 
                "colsample_bylevel": 0.8587289580606655, 
                "colsample_bytree": 1.0, 
                "reg_alpha": 0.006256200095839825, 
                "reg_lambda": 1.458362583960815
            }
        }
    }

    automl_settings.update(automl_config)

    automl = AutoML()

    np.random.seed(123)
    automl.fit(X_train, y_train, **automl_settings)

    y_pred_proba = automl.predict_proba(X_test)[:, 1]

    decision_lambda = np.vectorize(lambda p: 1 if p > threshold else 0)

    y_pred = decision_lambda(y_pred_proba)

    print('\nLoans refused: ')
    print(str(y_pred.sum()) + ' (' + str(round(y_pred.sum() / y_pred.size * 100, 1)) + '%)') 

    y_pred_proba_bins = pd.qcut(y_pred_proba, 5, duplicates = 'drop')
    print('\ny_pred_proba distribution:')
    print(y_pred_proba_bins.value_counts())

    print(colored('\nBest validation loss: ' + str(-automl.best_loss), 'blue'))

    print('\n')
    print(classification_report(y_test, y_pred))
    
    print('Normalized confusion matrix:')
    print(norm_confusion_matrix(y_test, y_pred))
    print('\n')

    print('Best model: ')
    print(automl.model.estimator)

    X_test_df = pd.DataFrame(X_test, columns=extra['features'])
    X_test_df['y_pred'] = y_pred
    X_test_df['y_pred_proba'] = y_pred_proba
    X_test_df['y_test'] = y_test

    for k,v in extra.items():
        if k.startswith('test_'):
            X_test_df[k] = v

    X_test_df.to_csv('X_test-' + experiment_id + '-' + features_file, index=False)

    avg_profit_pred = np.vectorize(avg_profit_func)(y_pred, y_test).sum()
    profit_pred = np.vectorize(profit_func)(y_pred, extra['test_profit']).sum()
    profit_test = extra['test_profit'].sum()

    print('\nAvg predicted test profit: ' + locale.currency(int(avg_profit_pred)))
    print('Predicted test profit: ' + locale.currency(int(profit_pred)))
    print('Actual test profit: ' + locale.currency(int(profit_test)))

    plot_learning_curve(log_file_name, time_budget = automl_config['time_budget'])
    plot_feat_imp(automl)

if __name__ == '__main__':

    locale.setlocale(locale.LC_ALL, '')

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--features', 
                        help='name of the CSV file with selected features',
                        default = 'variables-0.1.csv')

    parser.add_argument('-e', '--experimentid', 
                        help='id for this experiment/model version',
                        default = 'exp001')

    parser.add_argument('-t', '--threshold',
                        help = 'threshold for classification',
                        default = 0.5,
                        type = float)

    args, other = parser.parse_known_args()

    features_file = args.features
    experiment_id = args.experimentid
    threshold = args.threshold

    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--time-budget', type=int)
    parser2.add_argument('--ensemble', action='store_true')
    args2 = parser2.parse_args(other)
    
    config = {}

    if args2.time_budget is not None:
        config['time_budget'] = args2.time_budget

    if args2.ensemble is not None:
        config['ensemble'] = args2.ensemble

    main(features_file, experiment_id, config, threshold)
