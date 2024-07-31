from queue import Empty
import uvicorn
from fastapi import FastAPI
from westgate.flaml_model_core import LendingModel, UWModel, EmptyDataFrameException
from westgate.flaml_model_utils import load_model
import numpy as np
import pandas as pd
from pydantic import BaseModel, EmailStr
from typing import Literal, List, Dict, Any
from datetime import date
import os
from supabase import create_client, Client

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
model_refusal:LendingModel = load_model('refusal', basefolder='model_binaries/')
model_default:UWModel = load_model('default', basefolder='model_binaries/')

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class Attributes(BaseModel):
    
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
    average_closing_balance_day_after_income: float
    average_micro_loan_payment: float
    average_monthly_free_cash_flow: float
    average_monthly_nsf_fees_count: float
    average_monthly_recurring_transfers_in_complex: float
    average_employer_income_deposit: float


class Transaction(BaseModel):
    Date: date
    Code: Any
    Description: str
    Debit: float|None
    Credit: float|None
    Balance: float|None
    Id: str

class Holder(BaseModel):
    Name: str
    Address: Dict
    Email: EmailStr
    PhoneNumber: str

class AccountInfo(BaseModel):
    Transactions: List[Transaction]
    TransitNumber: str|None
    InstitutionNumber: str|None
    Title: str|None
    AccountNumber: str|None
    LastFourDigits: str|None
    Balance: Dict
    Category: str
    Type: str
    Currency: str
    Holder: Holder
    Id: str

class AccountDetails(BaseModel):
    HttpStatusCode: int
    Accounts: List[AccountInfo]


@app.get('/')
def index():
    return {'message': 'Westgate UW'}

def save_decision(org, 
                  decision, 
                  info, 
                  refusal_score=None,
                  refusal_percentile=None,
                  default_score=None,
                  default_percentile=None):
    r = supabase.table('logs').insert({
                'organization': org,
                'decision': decision,
                'info': info,
                'refusal_score': refusal_score,
                'refusal_percentile': refusal_percentile,
                'default_score': default_score,
                'default_percentile': default_percentile,
            }).execute()
    return r

def save_attributes(data, decision_id):
    supabase.table('attributes').insert({
            'attributes_json': data,
            'decision_id': decision_id
        }).execute()

def save_trx(details:Attributes, decision_id):
    if (details is not None) and (details.HttpStatusCode==200) and (len(details.Accounts) > 0):
            print('Saving account holder information')
            supabase.table('account_holder').insert({
                'decision_id': decision_id,
                'name': details.Accounts[0].Holder.Name,
                'address': details.Accounts[0].Holder.Address,
                'email': details.Accounts[0].Holder.Email,
                'phone_number': details.Accounts[0].Holder.PhoneNumber
            }).execute()

def save_all(org, 
            decision, 
            info, 
            attributes=None,
            details=None,
            refusal_score=None,
            refusal_percentile=None,
            default_score=None,
            default_percentile=None):
    
    r = save_decision(org, decision, info, 
                      refusal_score, refusal_percentile,
                      default_score, default_percentile)

    attributes['dob'] = attributes['dob'].strftime('%Y-%m-%d')
    attributes['request_date'] = attributes['request_date'].strftime('%Y-%m-%d')

    save_attributes(attributes, r.data[0]['id'])
    save_trx(details, r.data[0]['id'])


@app.post('/predict')
def predict(attributes:Attributes, details:AccountDetails|None=None, org='westgate'):

    data = attributes.model_dump()
    df = pd.DataFrame.from_dict([data])

    auto_refuse = (
        (df['account_age_days'] < 85).bool()
        or (df['count_nsf_90_days'] > 8).bool()
        or (df['count_nsf_30_days'] > 6).bool()
        or (df['count_stop_payment_90_days'] > 4).bool()
    )

    if auto_refuse:
        result = {
            'uw_decision': 'refuse',
            'score': 1,
            'percentile': 100,
            'info': 'auto_refuse'
        }

        save_all(
            org,
            'refuse',
            'auto-refuse',
            data,
            details
        )

        return result

    else:
        try:
            pred_refusal = model_refusal.predict_proba(df, filter=True, engineer=True).item()
            pred_default = model_default.predict_proba(df, filter=True, engineer=True).item()

            score = 0.5 * (pred_refusal + pred_default)

            percentiles_refusal = list(model_refusal.percentiles.keys())
            percentiles_default = list(model_default.percentiles.keys())

            #refusal

            idx_refusal = np.digitize(pred_refusal, list(model_refusal.percentiles.values()))

            if idx_refusal < len(percentiles_refusal):
                percentile_refusal = list(model_refusal.percentiles.keys())[idx_refusal]
            else:
                percentile_refusal = 100

            #default

            idx_default = np.digitize(pred_default, list(model_default.percentiles.values()))

            if idx_default < len(percentiles_default):
                percentile_default = list(model_default.percentiles.keys())[idx_default]
            else:
                percentile_default = 100

            percentile = 0.5 * (percentile_refusal + percentile_default)

            if (pred_refusal >= model_refusal.percentiles[35]) or \
                (pred_default >= model_default.percentiles[80]):
                uw_decision = 'refuse'
            else:
                uw_decision = 'accept'

            result = {
                'uw_decision': uw_decision,
                'score': score,
                'percentile': percentile,
                'info': None
            }

            save_all(
                org,
                uw_decision,
                None,
                data,
                details,
                pred_refusal,
                percentile_refusal,
                pred_default,
                percentile_default
            )

            return result

        except EmptyDataFrameException as e:
            info = ['Empty dataframe after filtering',
                    'Possible causes: minimum balance was too low']
            r = save_decision(org=org, 
                              decision='n/a', 
                              info='Empty dataframe after filtering')
            
            return {
                'uw_decision': 'n/a',
                'score': 0,
                'percentile': 0,
                'info': info
            }
        
    
    

    # return {
    #     'refusal':{
    #         'score': pred_refusal,
    #         'percentile': percentile_refusal
    #     },
    #     'default': {
    #         'score': pred_default,
    #         'percentile': percentile_default
    #     },
    #     'uw_decision': uw_decision,
    #     'version': 'v1.0',
    #     'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # }
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
