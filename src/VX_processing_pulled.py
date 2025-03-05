from pathlib import Path
from settings import config

OUTPUT_DIR = config("OUTPUT_DIR")
DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")

from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

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


def calc_cb_spread(df_cds, df_bond):
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

    df_merge = df_bond.merge(df_cds, on=["date", FIRM_ID, "tenor"], how='left')


    def interpolate_parspread(df):
        
        for (date, firm), group in df.groupby(["date", FIRM_ID]):
            group = group.sort_values(by="tenor")
            
            known_tenors = group["tenor"][group["parspread"].notna()]
            known_spreads = group["parspread"][group["parspread"].notna()]

            missing_tenors = group["tenor"][group["parspread"].isna()]
            
            if len(known_tenors) > 1 and not missing_tenors.empty:
                
                spline = CubicSpline(known_tenors, known_spreads, bc_type='natural')
                interpolated_spreads = spline(missing_tenors)
                
                for tenor, spread in zip(missing_tenors, interpolated_spreads):
                    df.loc[(df["date"] == date) & (df["firm"] == firm) & (df["tenor"] == tenor), "parspread"] = spread
            else:
                # if the parspread can't be interpolated we just move on
                continue
        
        return df

    df_merge = interpolate_parspread(df_merge)

    df_merge = df_merge.dropna()
    
    # assume that the treasury spread is positive (corp - treas)
    # corporates will be discounted more heavily compared to treasuries so they should have higher yields
    df_merge['CB'] = df_merge['parspread'] - df_merge['treasury_spread']
    df_merge['treas_ytm'] = df_merge['offering_yield'] - df_merge['treasury_spread']
    df_merge['rfr'] = df_merge['treas_ytm'] - df_merge['CB']

    return df_merge

def generate_graph_rfr(df):
    '''
    assume input has original df_merge format
    we want to graph the rfr column
    '''
    

    return
    




                


