# %%
basefolder = 'C:/Users/karim/Documents/westgate/data/training/'

# %%
import pandas as pd
from westgate.flaml_model_trainer import UWModelTrainer
from westgate.flaml_model_core import UWModel
from westgate.flaml_model_utils import load_model
import pandas as pd
from westgate.combochart import combo_chart
import logging
import numpy as np

# %%
logger = logging.getLogger('westgate.flaml_model_trainer')
logging.basicConfig()
# fhandler = logging.FileHandler(filename='metrics.log', mode='w')
# formatter = logging.Formatter('%(message)s')
# fhandler.setFormatter(formatter)
# logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

# %%
default_model = UWModel()

default_model_trainer = UWModelTrainer(model_name='default', 
                                        model_core=default_model, 
                                        model_version='1.0',
                                        basefolder='default_model/')

# %%
data_2022_df = pd.read_csv(basefolder + 'loan_outcomes_2022.csv', encoding='latin')

# %%
jan2023_df = pd.read_csv(basefolder + 'loan_outcomes_jan2023.csv', encoding='latin')
feb2023_df = pd.read_csv(basefolder + 'loan_outcomes_feb2023.csv', encoding='latin')
mar2023_df = pd.read_csv(basefolder + 'loan_outcomes_mar2023.csv', encoding='latin')

# %%
accepted_df = pd.concat([data_2022_df, jan2023_df, feb2023_df, mar2023_df])

# %%
accepted_df['request_yyyymm'] = accepted_df['request_date'].str[0:7]
accepted_df['validation_flag'] = accepted_df['request_yyyymm'].isin(['2023-01', '2023-02'])
accepted_df['test_flag'] = accepted_df['request_yyyymm'].isin(['2023-03'])
accepted_df['weight'] = np.select(
    [
        accepted_df['request_yyyymm'] == '2022-01',
        accepted_df['request_yyyymm'] == '2022-02',
        accepted_df['request_yyyymm'] == '2022-03',
        accepted_df['request_yyyymm'] == '2022-04',
        accepted_df['request_yyyymm'] == '2022-05',
        accepted_df['request_yyyymm'] == '2022-06',
        accepted_df['request_yyyymm'] == '2022-07',
        accepted_df['request_yyyymm'] == '2022-08',
        accepted_df['request_yyyymm'] == '2022-09',
        accepted_df['request_yyyymm'] == '2022-10',
        accepted_df['request_yyyymm'] == '2022-11',
        accepted_df['request_yyyymm'] == '2022-12',
        accepted_df['request_yyyymm'] == '2023-01',
        accepted_df['request_yyyymm'] == '2023-02',
        accepted_df['request_yyyymm'] == '2023-03'
    ],
    [
        1/12,
        2/12,
        3/12,
        4/12,
        5/12,
        6/12,
        7/12,
        8/12,
        9/12,
        10/12,
        11/12,
        1,
        1,
        1,
        1
    ],
    default=1
)


# %%
accepted_df['profit'] = accepted_df['total_paid'] - accepted_df['principal']

# %%
accepted_df[default_model.target] = np.where(accepted_df['profit'] < 50, 1, 0)

# %%
filtered_df = default_model.filter_df(accepted_df)

# %%
# here X_teste is December

X_train, X_test, y_train, y_test, extra = \
    default_model_trainer.split_data(filtered_df, 
                                    split_criteria='test_flag',
                                    extra_cols=['weight', 'validation_flag', 'request_yyyymm'])

X_train, X_test = default_model_trainer.feature_engineer(X_train, X_test)

#TODO: deal with na values in X_train/X_test if any
#non_na_idx = X['age'].notna()

#X = X[non_na_idx]

#if y is not None:
    #y = y[non_na_idx]

# %%
X_val, y_val = X_train.loc[extra['train_validation_flag']], y_train.loc[extra['train_validation_flag']]
X_train, y_train = X_train.loc[~extra['train_validation_flag']], y_train.loc[~extra['train_validation_flag']]

# %%
assert (len(X_train) + len(X_val) + len(X_test) == len(filtered_df))

# %%
weight_train = extra['train_weight'].loc[~extra['train_validation_flag']]
weight_val = extra['train_weight'].loc[extra['train_validation_flag']]
weight_test = extra['test_weight']

# %%
def p90_accuracy_metric(X_val, y_val, estimator, labels,
                X_train, y_train, weight_val=None, weight_train=None,
                *args):
    y_probas = estimator.predict_proba(X_val)[:, 1]
    df = pd.DataFrame({'proba': y_probas, 'y_val': y_val})
    perc90 = np.percentile(y_probas, 90)
    perc10 = np.percentile(y_probas, 10)
    delta = df.loc[df['proba'] >= perc90]['y_val'].sum() - df.loc[df['proba'] <= perc10]['y_val'].sum()
    return -delta, {}
        
config = {
    #"ensemble": True,
    "estimator_list": ['xgb_limitdepth'],
    #"estimator_list": ['xgboost'],
    "metric": "roc_auc",
    #"metric": p90_accuracy_metric,
    "eval_method": 'holdout',
    'X_val': X_val,
    'y_val': y_val,
    'sample_weight': weight_train,
    'sample_weight_val': weight_val
}

y_pred_proba, y_pred, extra = default_model_trainer.fit(
        X_train, X_test, y_train, y_test, extra,
        time_budget=180, 
        automl_config=config,
        show_plots=True,
        percentile=90
)

# %%
Xfull = pd.concat([X_train, X_val, X_test])

# %%
yfull = pd.concat([y_train, y_val, y_test])

# %%
weight_full = pd.concat([weight_train, weight_val, weight_test])

# %%
default_model_trainer.retrain_full(
    Xfull,
    yfull,
    weight_full=weight_full,
    time_budget=60
)

# %%
for handler in logger.handlers:
    logger.removeHandler(handler)
    handler.close()


