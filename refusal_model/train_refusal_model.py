
# %%
from westgate.flaml_model_trainer import LendingModelTrainer
from westgate.flaml_model_core import LendingModel, UWModel
from westgate.flaml_model_utils import load_model
import pandas as pd
from westgate.combochart import combo_chart
import os
import logging
import mlflow
import argparse
import json
import tempfile

# to view MLFlow runs (from default_model folder): 
# mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db


def main(time):

    logger = logging.getLogger('westgate.flaml_model')
    fhandler = logging.FileHandler(filename='metrics.log', mode='w')
    formatter = logging.Formatter('%(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlruns.db'
    mlflow.set_experiment("Refusal model")

    base_folder = 'C:/Users/karim/Documents/westgate'
    data_folder = base_folder + '/data/training'

    default_model = load_model(model_name='default', basefolder=f'{base_folder}/default_model/')
    refusal_model = LendingModel()

    refusal_model_trainer = LendingModelTrainer(model_name='refusal', 
                                            model_core=refusal_model, 
                                            model_version='1.0',
                                            basefolder=f'{base_folder}/refusal_model/')

    # %%
    accepted_2022_df = pd.read_csv(data_folder + '/loan_outcomes_2022.csv', encoding='latin')
    refused_2022_df = pd.read_csv(data_folder + '/refusal2022.csv', encoding='latin')

    # %%
    refused_2022_df['Id'] = refused_2022_df['loginId']

    # %%
    accepted_2023_df = pd.read_csv(data_folder + '/janToNov2023LoansAccepted.csv')
    refused_2023_df = pd.read_csv(data_folder + '/refusal2023_IF.csv')

    # %%
    refused_2023_df['Id'] = refused_2023_df['loginId']

    # %%
    accepted_2023_df['gender'] = accepted_2023_df['gender'].str.lower()
    accepted_2023_df['Id'] = accepted_2023_df['LoanId']

    # %%
    refused_2023_df.rename(columns={'requestDate': 'request_date'}, inplace=True)

    # %%
    refused_2024_IF_df = pd.read_csv(data_folder + '/refusal2024_IF.csv')
    refused_2024_PR_df = pd.read_csv(data_folder + '/refusal2024_PR.csv', encoding='latin')

    refused_2024_df = pd.concat([refused_2024_IF_df, refused_2024_PR_df])

    # %%
    refused_2024_df['Id'] = refused_2024_df['loginId']

    # %%
    refused_2024_df.rename(columns={'requestDate': 'request_date'}, inplace=True)

    # %%
    accepted_df = pd.concat([accepted_2022_df, accepted_2023_df])

    # %%
    refused_df = pd.concat([refused_2022_df, refused_2023_df, refused_2024_df])

    # %%
    refused_df['recurring_deposits_90_days'] = (
        refused_df['recurring_deposits_current_month'] + 
        refused_df['recurring_deposits_previous_month'] +
        refused_df['recurring_deposits_2_months_ago']
    )

    refused_df['sum_micro_loans_60_days'] = (
        refused_df['sum_micro_loan_payments_current_month'] + 
        refused_df['sum_micro_loan_payments_previous_month'] 
    )

    refused_df['recurring_deposits_90_days'] = (
        refused_df['recurring_deposits_current_month'] + 
        refused_df['recurring_deposits_previous_month'] +
        refused_df['recurring_deposits_2_months_ago']
    )

    refused_df['sum_micro_loans_60_days'] = (
        refused_df['sum_micro_loan_payments_current_month'] + 
        refused_df['sum_micro_loan_payments_previous_month'] 
    )

    # %%
    print('# loans before auto-refusal: ' + str(len(refused_df)))

    refused_df = refused_df[refused_df['account_age_days'] >= 85]

    refused_df = refused_df[refused_df['count_nsf_90_days'] <= 8]

    refused_df = refused_df[refused_df['count_nsf_30_days'] <= 6]

    refused_df = refused_df[refused_df['count_stop_payment_90_days'] <= 4]

    #refused_df = refused_df[refused_df['recurring_deposits_90_days']/3.0 >= 1800] <-- filers way too much

    #refused_df = refused_df[refused_df['sum_micro_loans_60_days'] <= 1000]

    print('# loans after auto-refusal: ' + str(len(refused_df)))

    # %%
    accepted_filtered_df = default_model.filter_df(accepted_df)

    # %%
    accepted_filtered_df['refused'] = 0
    refused_df['refused'] = 1

    df = pd.concat([accepted_filtered_df, refused_df])

    # %%
    X_train, X_test, y_train, y_test, extra = refusal_model_trainer.split_data(df, split_criteria=0.15)

    # %%
    X_train, X_test = refusal_model_trainer.feature_engineer(X_train, X_test)

    # %%
    filter = ~X_train['age'].isna()
    X_train = X_train.loc[filter]
    y_train = y_train.loc[filter]

    # %%
    filter = ~X_test['age'].isna()
    X_test = X_test.loc[filter]
    y_test = y_test.loc[filter]

    # %%
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # %%
    r = X_train, y_train, X_test, y_test, extra

    with mlflow.start_run():
            
        y_pred_proba, y_pred, extra = refusal_model_trainer.fit(*r, 
                                                        time_budget=time, 
                                                        threshold=0.55)

        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])

        refusal_model_trainer.retrain_full(
            X_full,
            y_full,
            time_budget=60
        )

        with open('refusal_model/refusal_percentiles.txt', 'w') as f:
            json.dump(refusal_model_trainer.model_core.percentiles, f)
            f.seek(0) 
            mlflow.log_artifact(f.name)

    # %%
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', default=600, type=int)
    args = parser.parse_args()

    time = args.time
    main(time)
