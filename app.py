from queue import Empty
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from westgate.flaml_model_core import LendingModel, UWModel, EmptyDataFrameException
from westgate.flaml_model_utils import load_model
from westgate.auto_refuse import detect_pattern
import numpy as np
import pandas as pd
from pydantic import BaseModel, EmailStr
from typing import Literal, List, Dict, Any
from datetime import date
import os
from supabase import create_client, Client
import logging
import sys
import re
from datetime import timedelta, datetime
#from blomp_api import Blomp
import tempfile
import uuid
import pathlib


logger = logging.getLogger("westgate")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)
logger.info("App started")

#######################################
#--- Uncomment this for local testing
from dotenv import load_dotenv
load_dotenv()
#---

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

class AccountTransactions(BaseModel):
    HttpStatusCode: int
    Accounts: List[AccountInfo]



def save_to_blomp(transactions:AccountTransactions, decision_id):

    email = os.environ["BLOMP_EMAIL"]
    password = os.environ["BLOMP_PASSWORD"]

    blomp = Blomp(email, password)

    # Get your cloud root directory
    root = blomp.get_root_directory()
    westgate_folder = root.get_folder_by_name('westgate')

    if westgate_folder is not None:

        current_month = datetime.today().strftime('%Y-%m')
        current_month_folder = root.get_folder_by_name(current_month)

        if current_month_folder is None:
            current_month_folder = westgate_folder.create_folder(current_month)

        if current_month_folder is not None:
            filepath = 'trx_' + str(decision_id) + '.txt'

            with open(filepath, 'w') as f:
                f.write(transactions.model_dump_json(indent=2))

            thread, _ = current_month_folder.upload(filepath)
            thread.join()
            pathlib.Path(filepath).unlink()


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
    logger.info(f"Saving UW decision: {decision}")
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
    logger.info(f"Saving attributes for decision_id {decision_id}")
    supabase.table('attributes').insert({
            'attributes_json': data,
            'decision_id': decision_id
        }).execute()

def save_trx(transactions:AccountTransactions, decision_id):
    logger.info(f"Saving transactions for decision_id {decision_id}")
    if (transactions is not None) and (transactions.HttpStatusCode==200) and (len(transactions.Accounts) > 0):
            print('Saving account holder information')
            supabase.table('account_holder').insert({
                'decision_id': decision_id,
                'name': transactions.Accounts[0].Holder.Name,
                'address': transactions.Accounts[0].Holder.Address,
                'email': transactions.Accounts[0].Holder.Email,
                'phone_number': transactions.Accounts[0].Holder.PhoneNumber
            }).execute()

def save_all(org, 
            decision, 
            info, 
            attributes=None,
            transactions=None,
            refusal_score=None,
            refusal_percentile=None,
            default_score=None,
            default_percentile=None):
    
    r = save_decision(org, decision, info, 
                      refusal_score, refusal_percentile,
                      default_score, default_percentile)
    
    decision_id = r.data[0]['id']

    attributes['dob'] = attributes['dob'].strftime('%Y-%m-%d')
    attributes['request_date'] = attributes['request_date'].strftime('%Y-%m-%d')

    save_attributes(attributes, decision_id)
    save_trx(transactions, decision_id)

    return decision_id


def compute_auto_refuse(transactions:AccountTransactions|None, request_date:date):
    # see: https://docs.google.com/spreadsheets/d/1ny6ntVl4rpk12ptQwzEW0AQ6RxfqmqolppWz48GLY7s/edit?gid=0#gid=0
    if transactions is None:
        print("Transactions is None")
        return False
    
    stop_payments_90d = 0
    nsf_90d = 0
    nsf_30d = 0
    lawyer_90d = 0

    for account in transactions.Accounts:
        for transaction in account.Transactions:
            cutoff_90d = request_date - timedelta(days=90)
            cutoff_30d = request_date - timedelta(days=30)
            if transaction.Date >= cutoff_90d:
                if detect_pattern('stop_payments', 
                                  transaction.Description,
                                  transaction.Debit,
                                  transaction.Credit):
                    stop_payments_90d += 1

                elif detect_pattern('lawyer', 
                                    transaction.Description,
                                    transaction.Debit,
                                    transaction.Credit):
                    lawyer_90d += 1

                elif detect_pattern('nsf',
                                    transaction.Description,
                                    transaction.Debit,
                                    transaction.Credit):
                    nsf_90d += 1
                    if transaction.Date >= cutoff_30d:
                        nsf_30d += 1

    print(f"COMPUTED STOP PAYMENTS 90d: {stop_payments_90d}")
    print(f"COMPUTED NSFs 90d: {nsf_90d}")
    print(f"COMPUTED NSFs 30d: {nsf_30d}")
    print(f"COMPUTED LAWYER 90d: {lawyer_90d}")

    if ((stop_payments_90d >= 4) 
        or (nsf_90d >= 8) 
        or (nsf_30d >= 6)
        or (lawyer_90d >= 1)):
        print("!! AUTOREFUSE - TRUE !!")
        return True
    else:
        return False


@app.post('/predict')
def predict(attributes:Attributes,
            transactions:AccountTransactions|None=None,
            org='westgate'):
    
    # Fetch latest thresholds from Supabase
    thresholds_resp = supabase.table('thresholds').select('*').order('created_at', desc=True).limit(1).execute()
    if not thresholds_resp.data or len(thresholds_resp.data) == 0:
        raise RuntimeError("No threshold record found in Supabase table 'thresholds'")
    latest_thresholds = thresholds_resp.data[0]
    refusal_threshold = int(latest_thresholds.get('refusal_threshold') * 100)
    default_threshold = int(latest_thresholds.get('default_threshold') * 100)
    if refusal_threshold is None or default_threshold is None:
        raise RuntimeError("Threshold record missing required fields")
    
    logger.info(f"Refusal threshold: {refusal_threshold}")
    logger.info(f"Default threshold: {default_threshold}")

    print(f"Org: {org}")

    data = attributes.model_dump()
    df = pd.DataFrame.from_dict([data])

    auto_refuse_basic = (
        (df['account_age_days'] < 85).bool()
        or (df['count_nsf_90_days'] > 8).bool()
        or (df['count_nsf_30_days'] > 6).bool()
        or (df['count_stop_payment_90_days'] > 4).bool()
    )

    auto_refuse_computed = compute_auto_refuse(transactions, attributes.request_date)

    auto_refuse = auto_refuse_basic or auto_refuse_computed

    if auto_refuse:
        result = {
            'uw_decision': 'refuse',
            'score': 1,
            'percentile': 100,
            'info': 'auto_refuse'
        }

        decision_id = save_all(
            org,
            'refuse',
            'auto-refuse',
            data,
            transactions
        )

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

            if (pred_refusal >= model_refusal.percentiles[refusal_threshold]) or \
                (pred_default >= model_default.percentiles[default_threshold]):
                uw_decision = 'refuse'
            else:
                uw_decision = 'accept'

            result = {
                'uw_decision': uw_decision,
                'score': score,
                'percentile': percentile,
                'info': None
            }

            decision_id = save_all(
                org,
                uw_decision,
                None,
                data,
                transactions,
                pred_refusal,
                percentile_refusal,
                pred_default,
                percentile_default
            )

        except EmptyDataFrameException as e:
            info = ['Empty dataframe after filtering',
                    'Possible causes: minimum balance was too low']
            
            r = save_decision(org=org, 
                              decision='n/a', 
                              info='Empty dataframe after filtering')
            decision_id = r.data[0]['id']
            
            return {
                'uw_decision': 'n/a',
                'score': 0,
                'percentile': 0,
                'info': info
            }
        
    # if transactions is not None:
    #     background_tasks.add_task(save_to_blomp, transactions, decision_id)

    return result
        
    
    

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
