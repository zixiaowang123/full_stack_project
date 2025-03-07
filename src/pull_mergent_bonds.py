"""
This scripts pulls the Mergent bond data from WRDS.
Code by adjusted by Alex Wang
"""

# Add src directory to Python path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import os

import pandas as pd
import wrds

from settings import config

# Set SUBFOLDER to the folder containing this file
SUBFOLDER = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = pd.Timestamp("1925-01-01")
END_DATE = pd.Timestamp("2024-01-01")


def get_bond_data_as_dict(wrds_username=WRDS_USERNAME):
    """
    Connects to a WRDS (Wharton Research Data Services) database and fetches bond data from Mergent
    on the the table named `fisd_fisd`. The data fetched includes the date,
    ticker, and parspread where the tenor is '5Y' and the country is 'United States'. The fetched data for each
    year is stored in a dictionary with the year as the key. The function finally returns this dictionary.

    Returns:
        dict: A dictionary where each key is a year from 2001 to 2023 and each value is a DataFrame containing
        the date, ticker, and parspread for that year.
    """
    db = wrds.Connection(wrds_username=wrds_username)
    bond_data = {}

    table_name = "fisd_fisd.fisd_mergedissue"
    query = f"""
    SELECT
        offering_date, -- The date the issue was originally offered.
        maturity, -- Date that the issue's principal is due for repayment.
        amount_outstanding, -- The amount of the issue remaining outstanding.
        security_level, -- SENS is Senior Unsecured Debt i think
        offering_yield, -- Yield to maturity at the time of issuance.
        issuer_cusip, -- issuer CUSIP.
        issue_cusip,
        treasury_spread -- The difference between the yield of the benchmark treasury issue and the issue's offering yield expressed in basis points.

    FROM
        {table_name} AS a
    WHERE
        (a.foreign_currency = 'N') AND
        (a.putable = 'N') AND
        (a.convertible = 'N') AND
        (a.isin != '') AND
        (a.canadian = 'N') AND
        (a.action_price > 0.5)
    """
    bond_data = db.raw_sql(query, date_cols=["date"])
    return bond_data

def filter_data(data):
    """
    Filters the bonds based on the required filters in the paper
    """
    bonds = data[data["amount_outstanding"] >= 10000]
    bonds.loc[:,"maturity_time_frame"] = bonds.loc[:,"maturity"] - bonds.loc[:,"offering_date"]
    bonds.loc[:,"maturity_time_frame"] = bonds.loc[:,"maturity_time_frame"].astype(str).str.split(',').str[0]
    bonds.loc[:,"maturity_time_frame"] = bonds.loc[:,"maturity_time_frame"].astype(str).str.split(' ').str[0]
    bonds.loc[:,"maturity_time_frame"] = pd.to_numeric(bonds.loc[:,"maturity_time_frame"], errors='coerce') / 365.25
    bonds = bonds.dropna(subset=['maturity_time_frame'])
    bonds['maturity_time_frame'] = bonds['maturity_time_frame'].round().astype(int)
    
    mask_g1 = bonds["maturity_time_frame"] >= 1
    mask_l10 = bonds['maturity_time_frame'] <= 10

    return bonds[mask_g1 & mask_l10]

if __name__ == "__main__":
    combined_df = get_bond_data_as_dict(wrds_username=WRDS_USERNAME)
    filered = filter_data(combined_df)
    (DATA_DIR).mkdir(parents=True, exist_ok=True)
    filered.to_parquet(DATA_DIR / "mergent_bond.parquet")
