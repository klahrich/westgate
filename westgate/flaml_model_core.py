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
from westgate.flaml_model_utils import load_model, feature_engineer_basic, EmptyDataFrameException
from sklearn.model_selection import ShuffleSplit
import logging
import dill

pd.set_option('mode.chained_assignment', None)
locale.setlocale(locale.LC_ALL, '')


# in the features.csv file, 
# variables marked 'extra' are required for feature engineering and saved in features_in
# variables mared 'optional' are optional and not saved in features_in
# both are saved in the extra dict object

class LendingModelBasic:

    def __init__(self, feature_file:str):
        self.features_df = self.read_feature_file(feature_file)
        self.features_in = self.set_features_in()
        self.automl = None
        self.features_in = None
        self.features = None
        self.feature_engs:List[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self.add_feature_eng(feature_engineer_basic)
        self.thresholds_min = None
        self.thresholds_max = None
        self.percentiles = None

    
    def read_feature_file(self, feature_file:str):
        features_df = pd.read_csv(feature_file)

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

    def set_features_in(self):
        return [c for c in list(self.features_df['variable']) 
                if (c not in list(self.optional_features))
                and (c != self.target)]

    def set_features(self, X_train):
        return [c for c in X_train.columns if 
                (c not in list(self.extra_features))
                and (c not in list(self.optional_features))]

    def save(self):
        with open(str(self.model_name) + '_model.dill', 'wb') as file:
            dill.dump(self, file)

    def add_feature_eng(self, fe:Callable[[pd.DataFrame], pd.DataFrame]):
        self.feature_engs.append(fe)

    def filter_df(self, original_df):

        df = original_df.copy()

        bfr = len(df)

        for r in self.thresholds_min:
            df = df[~(df[r['variable']] < r['threshold_min'])]

        for r in self.thresholds_max:
            df = df[~(df[r['variable']] > r['threshold_max'])]

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

    def predict_proba(self, df, filter=True, engineer=True):

        if filter:
            df = self.filter_df(df)
            if len(df) == 0:
                raise EmptyDataFrameException('No rows remaining after filtering.')

        if engineer:
            df = self.feature_engineer(df)
            if len(df) == 0:
                raise EmptyDataFrameException('No rows remaining after feature engineering.')

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

    def _feature_engineer(self, X):
        
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


class LendingModel(LendingModelBasic):
    
    def __init__(self, feature_file:str):
        super().__init__(feature_file)

    
    def _feature_engineer(self, X):
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

