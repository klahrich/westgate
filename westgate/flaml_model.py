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
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import date
from colored import Fore, Back, Style
from sklearn.ensemble._stacking import StackingClassifier
from westgate.combochart import combo_chart
from sklearn.model_selection import ShuffleSplit
import logging
import dill

pd.set_option('mode.chained_assignment', None)
locale.setlocale(locale.LC_ALL, '')


# in the features.csv file, 
# variables marked 'extra' are required for feature engineering and saved in features_in
# variables mared 'optional' are optional and not saved in features_in
# both are saved in the extra dict object


def load_model(model_name:str, basefolder='./'):
    with open(basefolder + model_name + '_model.dill', 'rb') as f:
        return dill.load(f)

def precision1(y_pred, y_val):
    if (positives := (y_pred == 1).sum()) > 0:
        precision_1 = ((y_pred == 1) & (y_val == 1)).sum() / positives
    else:
        precision_1 = 0
    return precision_1

def recall1(y_pred, y_val):  
    if (positives := (y_val == 1).sum()) > 0:
        recall_1 = ((y_pred == 1) & (y_val == 1)).sum() / (y_val == 1).sum()
    else:
        recall_1 = 0
    return recall_1

def fscore(precision_1, recall_1, beta=1):
    if precision_1==0 and recall_1==0:
        return 0
    else:
        return ((1+beta**2) * precision_1 * recall_1) / ((beta**2)*precision_1 + recall_1)


class EmptyDataFrameException(Exception):
    pass

class LendingModelBase:

    def __init__(self, model_name:str, model_version:str, basefolder='./', ylabel=''):
        self.model_name = model_name
        self.model_version = model_version
        self.log_file = basefolder + 'log_' + self.model_name + '.log'
        self.feature_file = basefolder + 'features_' + self.model_name + '.csv'
        self.features_df = self.read_feature_file()
        self.features_in = self.set_features_in()
        self.features = None
        self.automl = None
        self.basefolder = basefolder
        self.ylabel = ylabel
        self.percentiles = None

    def save(self):
        with open(str(self.model_name) + '_model.dill', 'wb') as file:
            dill.dump(self, file)
        
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

    def read_feature_file(self):
        features_df = pd.read_csv(self.feature_file)

        thresholds_min_df = features_df.loc[~features_df.threshold_min.isna(), ['variable', 'threshold_min']]
        self.thresholds_min = thresholds_min_df.to_dict(orient='records')

        thresholds_max_df = features_df.loc[~features_df.threshold_max.isna(), ['variable', 'threshold_max']]
        self.thresholds_max = thresholds_max_df.to_dict(orient='records')

        target = features_df.loc[features_df.type=='target', 'variable']
        target = target.item()
        self.target = target

        self.extra_features = features_df.loc[features_df.type=='extra', 'variable']
        self.optional_features = features_df.loc[features_df.type=='optional', 'variable']

        return features_df

    def filter_df(self, original_df):
        df = original_df.copy()

        for r in self.thresholds_min:
            df = df[~(df[r['variable']] < r['threshold_min'])]

        for r in self.thresholds_max:
            df = df[~(df[r['variable']] > r['threshold_max'])]
            
        return df

    def predict_proba(self, df, filter=True, engineer=True):

        if filter:
            df = self.filter_df(df)
            if len(df) == 0:
                raise EmptyDataFrameException('No rows remaining after filtering.')

        if engineer:
            df = self.feature_engineer(df)
            if len(df) == 0:
                raise EmptyDataFrameException('No rows remaining after feature engineering.')

        #prediction_df = df[self.features].to_numpy()
        preds =  self.automl.predict_proba(df[self.features])

        if preds.shape[0] > 1:
            preds = preds[:, 1]
            return pd.DataFrame({'pred_proba': preds}).set_index(df.index)

        elif preds.shape[0] == 1:
            pred = preds[0, 1]
            return pred

        else:
            raise Exception('Wrong preds dimension')

    def predict(self, df):
        df = self.filter_df(df)
        df = self.feature_engineer(df)
        preds = self.automl.predict(df[self.features].to_numpy())
        return pd.DataFrame({'pred': preds}).set_index(df.index)

    def split_data(self, 
                df:pd.DataFrame, 
                split_criteria:Callable[[pd.DataFrame], pd.Series]|str|float,
                extra_cols=[]):

        original_df = df.copy()

        extra = {}

        df = df[[c for c in df.columns if c in list(self.features_df['variable'])]]

        # now do the splitting
        target = self.target
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

        #extra['features'] = X_train.columns

        extra = self.update_extra(extra, X_train, y_train, X_test, y_test)

        return X_train, X_test, y_train, y_test, extra 

    def set_features_in(self):
        return [c for c in list(self.features_df['variable']) 
                if (c not in list(self.optional_features))
                and (c != self.target)]

    def set_features(self, X_train):
        return [c for c in X_train.columns if 
                (c not in list(self.extra_features))
                and (c not in list(self.optional_features))]

    def _feature_engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        
        def time_diff(X):
            request_date = X['request_date']
            dob = X['dob']
            try:
                if type(request_date)==str:
                    request_date = request_date[:10]
                    request_date = parse(request_date)
                if type(dob)==str:
                    dob = dob[:10]
                    dob = parse(dob)
                return relativedelta(request_date, dob).years
            except Exception as e:
                print(f'Error calculating age with dob {dob} and request_date {request_date}')
                print(e)
        
        assert 'dob' in X.columns
        assert 'request_date' in X.columns

        X['age'] = X[['request_date', 'dob']].apply(time_diff, axis=1)

        return X

    def feature_engineer(self, X_train, X_test=None):

        X_train = self._feature_engineer(X_train)

        if self.features is None:
            self.features = self.set_features(X_train)

        X_train = X_train[self.features]

        if X_test is not None:
            X_test = self._feature_engineer(X_test)

            X_test = X_test[self.features]
            return X_train, X_test

        return X_train


    def fit(self, X_train, y_train, X_test=None, y_test=None, extra={}, 
            time_budget=10, automl_config={}, show_stats=True, 
            show_plots=True, save_test=True, save_model=True,
            threshold=None, percentile=None):
        
        logger = logging.getLogger(__name__)

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

        np.random.seed(123)
        automl.fit(X_train, y_train, **automl_settings)

        self.automl = automl

        if show_plots:
            self.plot_learning_curve(time_budget)

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

                logger.info('\n')
                logger.info(classification_report(y_test, y_pred))

                logger.info('Confusion matrix:')
                logger.info(confusion_matrix(y_test, y_pred))
                logger.info('\n')
                
                logger.info('Normalized confusion matrix:\n')

                logger.info('By true:')
                logger.info(confusion_matrix(y_test, y_pred, normalize='true'))
                logger.info('\n')

                logger.info('By pred:')
                logger.info(confusion_matrix(y_test, y_pred, normalize='pred'))
                logger.info('\n')
                
                logger.info('By all:')
                logger.info(confusion_matrix(y_test, y_pred, normalize='all'))
                logger.info('\n')

            #print('Best model: ')
            #print(automl.model.estimator)

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

    def plot_learning_curve(self, time_budget):
        time_history, best_valid_loss_history, _, _, _ = get_output_from_log(filename=self.log_file,
                                                                            time_budget=time_budget)
        plt.title("Learning Curve")
        plt.xlabel("Wall Clock Time (s)")
        plt.ylabel("Validation Accuracy")
        plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
        plt.savefig('learning_plot_' + self.model_name + '.png')
        plt.show()

    def feat_imp(self):
        if type(self.automl.model)==StackingClassifier:
            # fi_df = pd.DataFrame({'variable': self.features})
            df_list = []
            for i, estimator in enumerate(self.automl.model.estimators_):
                df_list.append(
                    pd.DataFrame(
                        {
                            'model': type(estimator),
                            'variable': estimator.feature_names_in_,
                            'imp': estimator.feature_importances_
                        }
                    ).sort_values('imp', ascending=False)
                )
            return pd.concat(df_list)
        else:
            return pd.DataFrame({
                'variable': self.automl.model.estimator.feature_names_in_,
                'imp': self.automl.model.estimator.feature_importances_
            }).sort_values('imp', ascending=False)


# Refusal model with more feature engineering
class LendingModel(LendingModelBase):

    def __init__(self, 
                model_name:str, 
                model_version:str,
                basefolder='./',
                ylabel='Refusal'):
        super().__init__(model_name, model_version, basefolder, ylabel)
        
    def _feature_engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super()._feature_engineer(X)
        X['non_classified_income_current_month'] = (X['sum_non_employer_income_current_month']
                                                - X['sum_government_income_current_month'])

        X['non_classified_income_previous_month'] = (X['sum_non_employer_income_previous_month']
                                                    - X['sum_government_income_previous_month'])

        X['non_classified_income_2_months_ago'] = (X['sum_non_employer_income_2_months_ago']
                                                    - X['sum_government_income_2_months_ago'])

        X['monthly_repayment_capacity'] = (X['sum_employer_income_current_month']
                                            - X['sum_loan_deposits_90_days'] / 3.0)

        return X


class UWModel(LendingModelBase):

    def __init__(self, 
                model_name:str, 
                model_version:str,
                default:int=1, 
                repay:int=0, 
                basefolder='./',
                ylabel='Default'):
        super().__init__(model_name, model_version, basefolder, ylabel)
        self.default = default
        self.repay = repay

    def filter_df(self, original_df):

        bfr = len(original_df)
        df = super().filter_df(original_df)
        after = len(df)

        print(f'{bfr - after} rows removed by threshold filtering')

        def _filter(df, criteria, message):
            tmp = len(df)
            df = df.loc[criteria]
            if len(df) < tmp:
                print(f'{tmp - len(df)}' + ' ' + message)
            return df

        if 'error' in df.columns:
            df = _filter(df, 
                        df['error'].isna(), 
                        "Rows with 'error' column not NA have been discarded.")

        if 'Id' in df.columns:
            df = _filter(df,
                        ~df['Id'].isna(),
                        "Rows with 'Id' column NA have been discarded.")

        if 'request_date' in df.columns:
            df = _filter(df,
                        ~df['request_date'].isna(),
                        "Rows with 'request_date' column NA have been discarded.")

        if 'account_age_days' in df.columns:
            df = _filter(df,
                        df['account_age_days'] > 0,
                        "Rows with 'account_age_days' column not positive have been discarded.")

        return df

    def non_features(self):
        return ['total_paid', 'profit'] + [self.target]

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
            show_plots=True, save_test=True, save_model=True,
            threshold=None, percentile=None):

        if X_test is not None:
        
            y_pred_proba, y_pred, _ = super().fit(X_train, X_test, y_train, y_test, extra, 
                                                time_budget, automl_config, show_stats, 
                                                show_plots, save_test, save_model, threshold, percentile)

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
                perf_df = pd.DataFrame({'y_pred': y_pred_proba, 'y_test': y_test, 'profit_test': extra['test_profit']})
                combo_chart(perf_df, xvar='y_pred', q=10, yvar='profit_test', 
                            savefile='perf_' + self.model_name + '.png')

            return y_pred_proba, y_pred, extra

        else:
            super().fit(X_train, X_test, y_train, y_test, extra, 
                        time_budget, automl_config, show_stats, 
                        show_plots, save_test, save_model, threshold, percentile)

