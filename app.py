import uvicorn
from fastapi import FastAPI
from westgate.flaml_model import *
import pandas as pd
from pydantic import BaseModel
from typing import Literal
from datetime import date

app = FastAPI()
model_refusal = load_model('refusal_0.2', basefolder='model_binaries/')
model_default = load_model('default_1.0', basefolder='model_binaries/')

class LoanRequest(BaseModel):
    
    #refusal
    gender: Literal['male', 'female']
    dob: date
    request_date: date
    account_age_days: float
    count_nsf_90_days: float
    count_nsf_30_days: float
    count_stop_payment_90_days: float
    sum_micro_loan_payments_current_month: float
    sum_micro_loan_payments_previous_month: float
    sum_micro_loan_payments_2_months_ago: float
    recurring_deposits_current_month: float
    recurring_deposits_previous_month: float
    recurring_deposits_2_months_ago: float
    count_active_days: float
    sum_total_income_current_month: float
    sum_total_income_previous_month: float
    sum_total_income_2_months_ago: float
    sum_loan_payments_current_month: float
    sum_loan_payments_previous_month: float
    sum_loan_payments_2_months_ago: float
    sum_loan_deposits_30_days: float
    sum_loan_deposits_60_days: float
    sum_loan_deposits_90_days: float
    sum_government_income_current_month: float
    sum_government_income_previous_month: float
    sum_government_income_2_months_ago: float
    sum_employment_insurance_income_current_month: float
    sum_employment_insurance_income_previous_month: float
    sum_employment_insurance_income_2_months_ago: float
    sum_employer_income_current_month: float
    sum_employer_income_previous_month: float
    sum_employer_income_2_months_ago: float
    sum_disability_income_current_month: float
    sum_disability_income_previous_month: float
    sum_disability_income_2_months_ago: float
    average_monthly_recurring_transfers_in_complex: float
    average_monthly_recurring_transfers_out_complex: float

    #default
    employer_income_frequency: str
    average_closing_balance_day_after_employer_income: float
    average_closing_balance_day_of_employer_income: float
    balance_90_days_ago: float
    balance_current: float
    balance_min: float
    count_loan_payments_current_month: int
    count_loan_payments_previous_month: int
    count_distinct_micro_lenders: int
    count_days_negative_balance_90_days: int
    sum_non_employer_income_current_month: float
    sum_non_employer_income_previous_month: float
    sum_non_employer_income_2_months_ago: float
    sum_non_employer_income: float
    average_closing_balance_day_of_income: float
    average_monthly_micro_loan_payments_complex: float
    average_monthly_utility_payments_complex: float
    telecom_payments_average: float
    other_loan_payments_average: float
    utility_payments_average: float




@app.get('/')
def index():
    return {'message': 'Westgate UW'}

@app.post('/predict')
def predict_proba(data:LoanRequest):
    data = data.dict()
    df = pd.DataFrame.from_dict([data])
    pred_refusal = model_refusal.predict_proba(df, filter=True, engineer=True)
    pred_default = model_default.predict_proba(df, filter=True, engineer=True)

    idx_refusal = np.digitize(0.3, list(model_refusal.percentiles.values()))
    percentile_refusal = list(model_refusal.percentiles.keys())[idx_refusal]

    idx_default = np.digitize(0.3, list(model_default.percentiles.values()))
    percentile_default = list(model_default.percentiles.keys())[idx_default]

    return {
        'refusal':{
            'score': pred_refusal.item(),
            'percentile': percentile_refusal
        },
        'default': {
            'score': pred_default.item(),
            'percentile': percentile_default
        }
    }
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)