import pandas as pd
import numpy as np
from flaml import AutoML
#from flaml.automl.ml import norm_confusion_matrix
from typing import Dict, List
from sklearn.metrics import classification_report, confusion_matrix
import locale
from flaml.automl.data import get_output_from_log
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from westgate.combochart import combo_chart
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
