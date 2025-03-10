import pandas as pd
import pytest
from settings import config
import pull_fred
DATA_DIR = config("DATA_DIR")


def merge_treasuries_into_bonds(bonds_df, treas_df, day_window=3):
    '''
    Output: a dataframe combining treasury yields of similar timeframe to a dataframe of bonds

    bonds_df: DF of bonds with columns
        offering_date, -- The date the issue was originally offered.
        company_symbol, -- Company Symbol (issuer stock ticker).
        maturity, -- Date that the issue's principal is due for repayment.
        principal_amt, -- face or par value of bond.
        amount_outstanding, -- The amount of the issue remaining outstanding.
        security_level, -- SEN is Senior Unsecured Debt i think
        date, -- Monthly Date
        yield, -- Yield to maturity at the time of issuance.
        cusip,
        isin,
        rating_class, -- Combined Rating Class: 0.IG or 1.HY
        conv, -- Flag Convertible (1 or 0)
        offering_price, -- Offering Price
        price_eom, -- Price-End of Month
        t_spread -- Avg Bid/Ask Spread
    
    treas_df: DF of treasuries with columns
        kycrspid, kytreasno: identification for treasuries
        mcaldt: date when the data was collected (Monthly Date)
        tmpubout:
        tmyld:
        tmatdt
    '''
