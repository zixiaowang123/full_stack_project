import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm


DATA_DIR = config("DATA_DIR")
TREASURY_ISSUE_FILE_NAME = "issue_data.parquet"
TREASURY_MONTHLY_FILE_NAME = "monthly_ts_data.parquet"
CORPORATES_MONTHLY_FILE_NAME = "wrds_bond.parquet"

def generate_treasury_data(issue_df, tm_df):
    '''
    issue_df: Dataframe containing treasury bond and issue data
        kycrspid: identification id
        kytreasno: identification id
        tmatdt: maturity date

    tm_df: Dataframe containing treasury bond monthly time series
        kycrspid: identification id
        kytreasno: identification id
        mcaldt: date of report
        tmpubout: face value of public outstanding
        tmyld: daily yield (daily discount rate)

    output: merge dataframes to contain maturity date and then annualize daily yield
    drop NaNs for things that are necessary for analysis later
    '''
    # merge in maturity
    t_df = tm_df.merge(issue_df, on=['kycrspid', 'kytreasno'], how='left')

    # dropna in necessary columns
    t_df = t_df.dropna(subset=['mcaldt', 'tmyld', 'tmatdt', 'kycrspid', 'kytreasno'])

    # annualize yields
    t_df['treas_yld'] = (1 + t_df['tmyld']) ** 365 - 1

    # remove the original column
    t_df = t_df.drop(columns={'tmyld'})

    return t_df




def merge_treasuries_into_bonds(bond_df, treas_df, day_window=3):
    '''
    Output: a dataframe combining treasury yields of similar timeframe to a dataframe of bonds

    bond_df: DF of bonds with columns
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
        rating, -- Combined Rating Class: 0.IG or 1.HY
        conv, -- Flag Convertible (1 or 0)
        offering_price, -- Offering Price
        price_eom, -- Price-End of Month
        t_spread -- Avg Bid/Ask Spread
    
    treas_df: DF of treasuries with columns
        kycrspid, kytreasno: identification for treasuries
        mcaldt: date when the data was collected (Monthly Date)
        tmpubout: face value of bonds of this type held by the public
        treas_yld: yield of the treasury (annualized)
        tmatdt: date when the treasury matures

    day_window: window to find successful matching for treasuries

    output: dataframe with treasury data merged into all valid ones for bond_df
    '''

    # assume all necessary filtering has been done on bond df
    
    # pre work for datetime functionality and filtering
    treas_df = treas_df.reset_index()
    bond_df = bond_df.reset_index()

    bond_df = bond_df[['cusip, company_symbol, date, maturity, amount_outstanding, yield, rating, price_eom, t_spread']]

    bond_df['date'] = pd.to_datetime(bond_df['date'])
    bond_df['maturity'] = pd.to_datetime(bond_df['maturity'])

    treas_df['mcaldt'] = pd.to_datetime(treas_df['mcaldt'])
    treas_df['tmatdt'] = pd.to_datetime(treas_df['tmatdt'])

    # get year-month values for the report dates
    # days are a little off so we do this instead
    bond_df['year_m'] = bond_df['date'].dt.to_period('M')
    treas_df['year_m'] = treas_df['mcaldt'].dt.to_period('M')
    
    ym_set = set(bond_df['year_m'].unique()) # set for filtering treasuries first

    treas_df = treas_df[treas_df['year_m'].isin(ym_set)] # treasury pre filtering

    treas_mat_set = set(treas_df['tmatdt'])
    maturity_dates = pd.Series(bond_df['maturities'].unique())
    valid_mats = maturity_dates.apply(
        lambda x: any(mat_date - timedelta(days=3) <= x <= mat_date + timedelta(days=3) for mat_date in treas_mat_set)
    )

    v_mat_set = set(maturity_dates[valid_mats])
    bond_df = bond_df[bond_df['maturity'].isin(v_mat_set)]

    treas_df['v_id'] = treas_df.index

    # Start merging
    
    # Initialize treasury dictionary for merging
    treas_dict = {y_m: treas_df[treas_df.year_m == y_m] for y_m in ym_set}

    lookup_cache = {}

    def get_v_id(year_m, maturity):
        """ Function to find the best-matching v_id for each unique (year_m, maturity) pair """
        if (year_m, maturity) in lookup_cache:
            return lookup_cache[(year_m, maturity)]

        t_df = treas_dict.get(year_m, pd.DataFrame()).copy()

        if t_df.empty or pd.isna(maturity):
            lookup_cache[(year_m, maturity)] = np.NaN
            return np.NaN

        t_df['tmatdt'] = pd.to_datetime(t_df['tmatdt'], errors='coerce')
        maturity_date = pd.to_datetime(maturity, errors='coerce')

        if t_df['tmatdt'].isna().all() or pd.isna(maturity_date):
            lookup_cache[(year_m, maturity)] = np.NaN
            return np.NaN

        t_df['day_diff'] = abs((t_df['tmatdt'] - maturity_date).dt.days)
        t_df = t_df[t_df['day_diff'] <= 3]

        if t_df.empty:
            lookup_cache[(year_m, maturity)] = np.NaN
            return np.NaN

        if t_df['tmpubout'].isna().all():
            lookup_cache[(year_m, maturity)] = t_df.iloc[0]['v_id']
            return t_df.iloc[0]['v_id']

        max_row_idx = t_df['tmpubout'].idxmax()
        lookup_cache[(year_m, maturity)] = t_df.loc[max_row_idx, 'v_id']
        return t_df.loc[max_row_idx, 'v_id']

    tqdm.pandas()
    bond_df['v_id'] = bond_df.progress_apply(lambda row: get_v_id(row.year_m, row.maturity), axis=1)

    merge_df = bond_df.merge(treas_df, on='v_id', how='left')

    return merge_df







    
