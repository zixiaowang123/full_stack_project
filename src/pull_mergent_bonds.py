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
        isin, -- The International Securities Identification Number associated with this issue.
        putable, -- Put option flag.
        convertible, -- Flag indicating the issue can be converted to the common stock (or other security).
        security_level -- SEN is Senior Unsecured Debt 

    FROM
        {table_name} AS a
    WHERE
        (a.foreign_currency = 'N') AND
        (a.putable = 'N') AND
        (a.convertible = 'N') AND
        (a.isin != '')
    """
    bond_data = db.raw_sql(query, date_cols=["date"])
    return bond_data


def combine_bond_data(cds_data: dict) -> pd.DataFrame:
    """
    Combines the bond data stored in a dictionary into a single DataFrame.

    For each key-value pair in `cds_data`, a new column "year" is added to the
    DataFrame containing the value of the key (i.e., the year). Then, all
    DataFrames are concatenated.

    Args:
        cds_data (dict): A dictionary where each key is a year and its value is
        a DataFrame with CDS data for that year.

    Returns:
        pd.DataFrame: A single concatenated DataFrame with an additional "year"
        column.
    """
    dataframes = []
    for year, df in cds_data.items():
        # Create a copy to avoid modifying the original DataFrame
        df_with_year = df.copy()
        df_with_year["year"] = year
        dataframes.append(df_with_year)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def pull_bond_data(wrds_username=WRDS_USERNAME):
    bond_data = get_bond_data_as_dict(wrds_username=wrds_username)
    combined_df = combine_bond_data(bond_data)
    return combined_df


def load_bond_data(data_dir=DATA_DIR, subfolder=SUBFOLDER):
    path = data_dir / subfolder / "mergent_bond.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    combined_df = get_bond_data_as_dict(wrds_username=WRDS_USERNAME)
    (DATA_DIR / SUBFOLDER).mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(DATA_DIR / SUBFOLDER / "mergent_bond.parquet")
