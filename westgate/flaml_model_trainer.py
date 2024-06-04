import pandas as pd
import numpy as np
from flaml import AutoML
#from flaml.automl.ml import norm_confusion_matrix
from typing import Dict, List, Callable
from sklearn.metrics import classification_report, confusion_matrix
import locale
from flaml.automl.data import get_output_from_log
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from dateutil.parser import parse
from colored import Fore, Back, Style
from westgate.combochart import combo_chart
from westgate.flaml_model_core import LendingModelCore
from sklearn.model_selection import ShuffleSplit
import logging
from dvclive import Live

pd.set_option('mode.chained_assignment', None)
locale.setlocale(locale.LC_ALL, '')


# in the features.csv file, 
# variables marked 'extra' are required for feature engineering and saved in features_in
# variables mared 'optional' are optional and not saved in features_in
# both are saved in the extra dict object

class LendingModelTrainer:

    def __init__(self, 
                 model_name:str, 
                 model_core:LendingModelCore, 
                 model_version:str, 
                 basefolder='./', 
                 ylabel=''):
        self.model_name = model_name
        self.model_core = model_core
        self.model_core.model_name = self.model_name
        self.features_file = basefolder + 'features_' + self.model_name + '.csv'
        self.model_core.set_features_file(self.features_file)
        self.model_version = str(model_version)
        self.log_file = basefolder + 'log_' + self.model_name + '.log'
        self.basefolder = basefolder
        self.ylabel = ylabel
        
    def update_extra(self, extra, X_train, y_train, X_test, y_test) -> Dict:
        for f in list(self.extra_features):
            extra['train_' + f] = X_train[f]
            extra['test_' + f] = X_test[f]

        for f in list(self.optional_features):
            if f in X_train.columns:
                extra['train_' + f] = X_train[f]
            if f in X_test.columns:
                extra['test_' + f] = X_test[f]

        return extra

    def split_data(self, 
                df:pd.DataFrame, 
                split_criteria:Callable[[pd.DataFrame], pd.Series]|str|float,
                extra_cols=[]):

        original_df = df.copy()

        extra = {}

        df = df[[c for c in df.columns if c in list(self.model_core.features_df['variable'])]]

        # now do the splitting
        target = self.model_core.target
        X = df[[c for c in df.columns if c != target]]
        y = df[target]

        if callable(split_criteria) or type(split_criteria)==str:
            if callable(split_criteria):
                v = split_criteria(X)
            
            elif type(split_criteria)==str:
                v = original_df[split_criteria]

            X_train = X.loc[~v, :]
            X_test = X.loc[v, :]
            y_train = y[~v]
            y_test = y[v]

            for col in extra_cols:
                extra['train_' + col] = original_df.loc[~v, col]
                extra['test_' + col] = original_df.loc[v, col]

        elif type(split_criteria)==float:
            splitter = ShuffleSplit(n_splits=1, test_size=split_criteria)
            train_idx, test_idx = list(splitter.split(X))[0]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for col in extra_cols:
                extra['train_' + col] = original_df.iloc[train_idx, col]
                extra['test_' + col] = original_df.iloc[test_idx, col]

        extra = self.update_extra(extra, X_train, y_train, X_test, y_test)

        return X_train, X_test, y_train, y_test, extra 

    def filter_df(self, original_df) -> pd.DataFrame:
        return self.model_core.filter_df(original_df)    

    def feature_engineer(self, X_train, X_test=None):
        return self.model_core.feature_engineer(X_train, X_test)

    def fit(self, X_train, y_train, X_test=None, y_test=None, extra={}, 
            time_budget=10, automl_config={}, show_stats=True, 
            show_plots=True, save_test=False, 
            threshold=None, percentile=None):
        
        logger = logging.getLogger('westgate.flaml_model_trainer')

        logger.info('Model: ' + self.model_name)
        logger.info('Model version: ' + self.model_version)
        logger.info('---\n')

        logger.info('X_train shape: ' + str(X_train.shape))
        logger.info('X_test shape: ' + str(X_test.shape))

        automl_settings = {
            "time_budget": time_budget,  # in seconds
            "metric": "roc_auc",
            "task": 'classification',
            "estimator_list": ['xgboost'],
            "log_file_name": self.log_file,
            "eval_method": "cv",
            "n_splits": 5,
            "retrain_full": True,
            "log_type": "all",
            "verbose": 2
        }

        automl_settings.update(automl_config)

        automl = AutoML()

        logger.info('Model training started...')
        np.random.seed(123)
        automl.fit(X_train, y_train, **automl_settings)

        self.model_core.automl = automl

        with Live() as live:

            if show_plots:
                self.plot_learning_curve(time_budget)
                live.log_image(self.model_name + '_learning_plot.png',
                               self.basefolder + self.model_name + '_learning_plot.png')

            if X_test is not None:

                assert ((threshold is not None) or (percentile is not None))
                assert ((threshold is None) or (percentile is None))

                y_pred_proba = automl.predict_proba(X_test)[:, 1]

                if threshold is None:
                    threshold = np.percentile(y_pred_proba, percentile)

                y_pred = np.where(y_pred_proba >= threshold, 1, 0)
        
                if show_stats:
                    logger.info('Actual ' + self.ylabel + '[TEST]:')
                    logger.info(str(y_test.sum()) + ' (' + str(round(y_test.sum() / y_test.size * 100, 1)) + '%)') 

                    logger.info('Predicted ' + self.ylabel + '[TEST]:')
                    logger.info(str(y_pred.sum()) + ' (' + str(round(y_pred.sum() / y_pred.size * 100, 1)) + '%)') 

                    y_pred_proba_df = pd.DataFrame({'y_pred_test': y_pred_proba})
                    y_pred_proba_df.plot.hist()

                    y_pred_proba_bins = pd.cut(y_pred_proba, 10, duplicates = 'drop')
                    logger.info('\ny_pred_proba distribution:')
                    logger.info(y_pred_proba_bins.value_counts())

                    logger.info('\nBest validation loss: ' + str(automl.best_loss))
                    live.log_metric(self.model_name + '_validation_loss', automl.best_loss)

                    logger.info('\n')
                    logger.info(classification_report(y_test, y_pred))
                    
                    cr = classification_report(y_test, y_pred, output_dict=True)
                    m0 = cr['0']
                    m1 = cr['1']
                    live.log_metric(self.model_name + '_0_precision', m0['precision'])
                    live.log_metric(self.model_name + '_0_recall', m0['recall'])
                    live.log_metric(self.model_name + '_0_f1-score', m0['f1-score'])
                    live.log_metric(self.model_name + '_1_precision', m1['precision'])
                    live.log_metric(self.model_name + '_1_recall', m1['recall'])
                    live.log_metric(self.model_name + '_1_f1-score', m1['f1-score'])

                    logger.info('Confusion matrix:')
                    logger.info(confusion_matrix(y_test, y_pred))
                    logger.info('\n')

                    perf_df = pd.DataFrame({'y_pred': y_pred_proba, 'y_test': y_test})
                    combo_chart(perf_df, xvar='y_pred', q=10, yvar='y_test', 
                                savefile=self.basefolder + self.model_name + '_perf.png')
                    live.log_image(self.model_name + '_perf.png',
                                    self.basefolder + self.model_name + '_perf.png')

                if save_test:
                    X_test_df = pd.DataFrame(X_test, columns=X_test.columns)
                    X_test_df['y_pred'] = y_pred
                    X_test_df['y_pred_proba'] = y_pred_proba
                    X_test_df['y_test'] = y_test

                    for k,v in extra.items():
                        if k.startswith('test_'):
                            X_test_df[k] = v.reset_index(drop=True)

                    print('Saving ' + 'X_test_' + self.model_name + '.csv')
                    X_test_df.to_csv(self.basefolder + 'X_test_' + self.model_name + '.csv', index=False)

            return y_pred_proba, y_pred, None

    def retrain_full(self, Xfull, yfull, weight_full=None, time_budget=60):
        self.model_core.automl.retrain_from_log(
            self.log_file,
            Xfull,
            yfull,
            sample_weight=weight_full,
            time_budget=time_budget,
            train_best=True,
            train_full=True,
        )
        preds_full = self.model_core.predict_proba(Xfull, filter=False, engineer=False)
        percentiles = np.percentile(preds_full['pred_proba'], range(5,100,5))
        self.model_core.percentiles = {p:v for p,v in zip(range(5,100,5), percentiles)}
        self.model_core.save(self.basefolder)

    def plot_learning_curve(self, time_budget):
        time_history, best_valid_loss_history, _, _, _ = get_output_from_log(filename=self.log_file,
                                                                            time_budget=time_budget)
        plt.title("Learning Curve")
        plt.xlabel("Wall Clock Time (s)")
        plt.ylabel("Validation Accuracy")
        plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
        plt.savefig(self.basefolder + self.model_name + '_learning_plot.png')

    def feat_imp(self):
        return self.model_core.feat_imp()


class UWModelTrainer(LendingModelTrainer):

    def __init__(self, 
                model_name:str, 
                model_core:LendingModelCore,
                model_version:str,
                default:int=1, 
                repay:int=0, 
                basefolder='./',
                ylabel='Default'):
        super().__init__(model_name, model_core, model_version, basefolder, ylabel)
        self.default = default
        self.repay = repay

    def update_extra(self, extra, X_train, y_train, X_test, y_test):
        extra['train_profit'] = X_train['total_paid'] - X_train['principal']
        extra['test_profit'] = X_test['total_paid'] - X_test['principal']

        return extra
    
    def avg_profit_func(self, y_pred, y_val): 
        return np.select(
            [(y_pred == self.repay) & (y_val == self.repay), 
            (y_pred == self.repay) & (y_val != self.repay),
            y_pred != self.repay],

            [[385], [-120], [0]]
        )

    def profit_func(self, y_pred, y_profit): 
        return np.where(y_pred == self.repay, y_profit, 0)

    def anti_profit_func(self, y_pred, y_profit):
        return np.where(y_pred != self.repay, y_profit, 0)

    def make_pr_metric(self, threshold, beta=1):

        def pr_metric(X_val, y_val, estimator, labels,
                        X_train, y_train, weight_val=None, weight_train=None,
                        *args):
            y_probas = estimator.predict_proba(X_val)[:, 1]

            y_pred = np.where(y_probas > threshold, 1, 0)

            if (positives := (y_pred == 1).sum()) > 0:
                precision_1 = ((y_pred == 1) & (y_val == 1)).sum() / positives
            else:
                precision_1 = 0

            if (positives := (y_val == 1).sum()) > 0:
                recall_1 = ((y_pred == 1) & (y_val == 1)).sum() / (y_val == 1).sum()
            else:
                recall_1 = 0

            if precision_1==0 and recall_1==0:
                return 0, {}
            else:
                return -((1+beta**2) * precision_1 * recall_1) / ((beta**2)*precision_1 + recall_1), {}

        return pr_metric

    def make_profit_metric(self, threshold):

        def profit_metric(X_val, y_val, estimator, labels,
                        X_train, y_train, weight_val=None, weight_train=None,
                        *args):

            y_probas = estimator.predict_proba(X_val)[:, 1]

            y_pred = np.where(y_probas > threshold, 1, 0)

            profit_pred = self.avg_profit_func(y_pred, y_val).sum()

            profit_actual = self.avg_profit_func(self.repay, y_val).sum()

            extra = {'profit_val_pred': int(profit_pred), 'profit_val_actual': int(profit_actual)}
            metric = (profit_actual - profit_pred) if not np.all(y_pred == 0) else 99999999

            return metric, extra

        return profit_metric

    def fit(self, X_train, y_train, X_test=None, y_test=None, extra=None, 
            time_budget=10, automl_config={}, show_stats=True, 
            show_plots=True, save_test=False, 
            threshold=None, percentile=None):

        if X_test is not None:
        
            y_pred_proba, y_pred, _ = super().fit(X_train, X_test, y_train, y_test, extra, 
                                                time_budget, automl_config, show_stats, 
                                                show_plots, save_test, threshold, percentile)

            predicted_profit_test = self.profit_func(y_pred, extra['test_profit']).sum()
            predicted_anti_profit_test = self.anti_profit_func(y_pred, extra['test_profit']).sum()
            predicted_avg_anti_profit_test = predicted_anti_profit_test / y_test.size
            actual_profit_test = extra['test_profit'].sum()

            extra['predicted_avg_anti_profit_test'] = predicted_avg_anti_profit_test

            delta = actual_profit_test - predicted_profit_test

            #print('\nAvg predicted test profit: ' + locale.currency(int(avg_profit_pred)))
            print('Profitability stats')
            print('----')
            print('Predicted profit [TEST]: ' + locale.currency(int(predicted_profit_test)))
            print('Actual profit [TEST]: ' + locale.currency(int(actual_profit_test)))
            print('Anti-profit [TEST]: ' + locale.currency(int(predicted_anti_profit_test)))

            color: str = f'{Style.underline}{Fore.cyan}{Back.black}'
            print(f'{color}Anti-profit per loan [TEST]: ' + \
                    locale.currency(int(predicted_avg_anti_profit_test)) + \
                    f'{Style.reset}')
            print('----')

            #assert abs(predicted_anti_profit_test/delta - 1) < 0.05

            if show_plots:
                with Live() as live:
                    perf_df = pd.DataFrame({'y_pred': y_pred_proba, 'y_test': y_test, 'profit_test': extra['test_profit']})
                    combo_chart(perf_df, xvar='y_pred', q=10, yvar='profit_test', 
                                savefile=self.basefolder + self.model_name +'_perf_uw.png')
                    live.log_image(self.model_name +'_perf_uw.png',
                                    self.basefolder + self.model_name + '_perf_uw.png')

            return y_pred_proba, y_pred, extra

        else:
            return super().fit(X_train, X_test, y_train, y_test, extra, 
                                time_budget, automl_config, show_stats, 
                                show_plots, save_test, threshold, percentile)

