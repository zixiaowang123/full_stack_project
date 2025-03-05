from pathlib import Path
from settings import config

OUTPUT_DIR = config("OUTPUT_DIR")
DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")

from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pull_public_repo_data

pull_public_repo_data.series_descriptions

# dataframe processing

FIRM_ID = 'firm_isin'

'''
formatting I need:

FR = ytm of corp bond with tenor tau - SOFR

SOFR is either the current value of SOFR or the 
swap rate with tenor tau for SOFR.

CDS = maturity matched CDS spread (parspread)

formula
CB = CDS - FR
'''

"""
additional comments:
for the dataframe of BONDS:
columns:
[date, offering_yield, treasury_spread, tenor, firm_isin]

for the dataframe of CDS:
[date, parspread, tenor, firm_isin]
"""

def calc_FR_values(df_bond):
    '''
    df_bond: dataframe of bonds with columns
    [date, offering_yield, treasury_spread, tenor, firm_isin]

    assuming treasury spread = treasury_yield - ytm

    output:
    a dataframe of the original bond data with the FR column and yield column added
    '''

    def calc_FR(row):
        mask1 = df_sofr.date == row.date
        if swap:
            mask2 = df_sofr.tenor == row.tenor
            try:
                return row.ytm - df_sofr[mask1 & mask2].iloc[0]['rate']
            except:
                print(f'There was no SOFR data for tenor {row.tenor} and date {row.date}.')
        else:
            try: 
                return row.ytm - df_sofr[mask1].iloc[0]['rate']
            except:
                print(f'There was no SOFR data for date {row.date}')

        return np.NaN
    
    df_bond['FR'] = df_bond.apply(lambda row: calc_FR(row), axis=1)

    return df_bond

def calc_cb_spread(df_cds, df_bond, df_treas):
    '''
    df_cds: dataframe of cds par spreads with columns
    [date, firm, parspread, tenor]
    df_bond: dataframe of fr spread
    [date, offering_yield, treasury_spread, tenor, firm_isin]

    output: a dataframe containing the parspread, fr, and the resulting CB spread
    defined as 
    CB = CDS - FR
    and the implied risk free arbitrage rate defined as
    rfr = abs(CB) - y

    rfr_arb is the final product
    '''

    df_merged = pd.merge(df_fr, df_cds, on=["date", "firm", "tenor"], how="inner")
    df_merged['CB'] = df_merged['parspread'] - df_merged['FR']

    df_treas.rename(columns={'ytm':'treas ytm'}, inplace=True)
    
    df_fin = pd.merge(df_merged, df_treas, on=['date', 'tenor'], how='inner')

    df_fin['rfr_arb'] = df_fin['CB'].abs() - df_fin['treas ytm']

    return df_fin
    




                


