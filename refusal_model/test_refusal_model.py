import numpy as np
import pandas as pd
from westgate.flaml_model_core import UWModel
from westgate.flaml_model_utils import load_model
from sklearn.metrics import classification_report
import argparse

base_folder = 'C:/Users/karim/Documents/westgate'
data_folder = base_folder + '/data/training'

def predict(model_refusal, df, percentile):

    pred_refusal = model_refusal.predict_proba(df, filter=True, engineer=True)

    df = df.loc[pred_refusal.index]

    y_test = np.where(df['profit'] < 50, 1, 0)

    y_pred = np.where(pred_default >= model_refusal.percentiles[percentile], 1, 0)

    df['y_test'] = y_test
    df['y_pred'] = y_pred

    df_denied = df[df['y_pred']==1]

    return {
        'nb_predicted_defaults': y_pred.sum(),
        'pct_predicted_defaults': y_pred.sum() / y_pred.size,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


def test(percentile, year:int, month:int):

    accepted_2023_df = pd.read_csv(data_folder + '/janToNov2023LoansAccepted.csv')
    refused_2023_df = pd.read_csv(data_folder + '/refusal2023_IF.csv')
    
    df = pd.read_csv(data_folder + '/loan_outcomes_' + month +'.csv', encoding='latin')

    model_default:UWModel = load_model('default', basefolder='model_binaries/')

    results = predict(model_default, df, percentile)

    return results

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--percentile', type=int, default=80)
    parser.add_argument('-m', '--month', type=str, default='jan2023')

    args = parser.parse_args()

    month = args.month
    percentile = args.percentile

    result = test(percentile, month)

    print(result)