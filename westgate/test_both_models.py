import numpy as np
import pandas as pd
from westgate.flaml_model_core import UWModel
from westgate.flaml_model_utils import load_model
from sklearn.metrics import classification_report
import argparse

base_folder = 'C:/Users/karim/Documents/westgate'
data_folder = base_folder + '/data/training'

def test(percentile, month:str):
    
    df = pd.read_csv(data_folder + '/loan_outcomes_' + month +'.csv', encoding='latin')

    model_default:UWModel = load_model('default', basefolder='model_binaries/')

    pred_default = model_default.predict_proba(df, filter=True, engineer=True)

    df = df.loc[pred_default.index]

    y_test = np.where(df['profit'] < 50, 1, 0)

    y_pred = np.where(pred_default >= model_default.percentiles[percentile], 1, 0)

    df['y_test'] = y_test
    df['y_pred'] = y_pred
    
    df.to_csv('data/test_default.csv')

    y_profit = df['profit']

    df_denied = df[df['y_pred']==1]

    print('Actual defaults' + '[TEST]:')
    print(str(y_test.sum()) + ' (' + str(round(y_test.sum() / y_test.size * 100, 1)) + '%)') 

    print('Predicted defaults' + '[TEST]:')
    print(str(y_pred.sum()) + ' (' + str(round(y_pred.sum() / y_pred.size * 100, 1)) + '%)') 

    print(classification_report(y_test, y_pred))

    print(f"Total anti-profit for denied loans: {df_denied['profit'].sum()}")
    print(f"Average anti-profit per denied loan: {df_denied['profit'].sum() / len(df_denied)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--percentile', type=int, default=80)
    parser.add_argument('-m', '--month', type=str, default='jan2023')

    args = parser.parse_args()

    month = args.month
    percentile = args.percentile

    test(percentile, month)