from abc import ABC, abstractmethod
import pandas as pd
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse


class FeatureEng(ABC):

    @abstractmethod
    def feature_engineer(self, X:pd.DataFrame) -> pd.DataFrame:
        pass


class BasicFeatureEng(FeatureEng):

    def feature_engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        
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
