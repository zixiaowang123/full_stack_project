from pathlib import Path
from settings import config
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

OUTPUT_DIR = config("OUTPUT_DIR")
DATA_DIR = config("DATA_DIR")

FINAL_ANALYSIS_FILE_NAME = "final_data.parquet"


def calc_cb_spread(df):
    '''
    INPUT WAS PREVIOUS FINAL PRODUCT
    df: dataframe with par spread values merged into all values where there was a possible cubic spline
       'cusip', -- unique bond tag
       'date', -- reporting date
       'maturity', -- maturity date of bond
       'yield', -- corporate bond yield
       'rating', -- rating: 1 for HY, 0 for IG
       'treas_yld', -- treasury yield for similar maturity and same report date
       'par_spread', -- parspread of CDS, backed out by Cubic Spline
       't_spread', -- bid ask spread on bond
       'price_eom', -- End Of Month Price of bond
       'amount_outstanding' -- face value outstanding

    output:
        additional columns:
        FR: Z-spread of the bond
            FR = yield - treas_yld
        CB: implied return on CDS-bond spread
            CB = par_spread - FR
        rfr: implied risk free rate
            rfr = treas_yld - CB
    '''
    df['FR'] = df['yield'] - df['treas_yld']
    df['CB'] = df['par_spread'] - df['FR']
    df['rfr'] = df['treas_yld'] - df['CB']

    df = df[df['rfr'] < 1] # remove unreasonable data, rfr is in absolute space

    return df

def generate_graph(df, col='rfr'):
    '''
    df from previous function with the additonal columns

    col is the value that is desired to be graphed
    '''
    df['date'] = pd.to_datetime(df['date'])


    # Compute the mean of the specified column per (date, rating) pair
    df_grouped = df.groupby(['date', 'rating'])[col].mean().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot for Investment Grade (IG) (rating = 0)
    df_ig = df_grouped[df_grouped['rating'] == 0]
    plt.plot(df_ig['date'], df_ig[col], label="IG (Investment Grade)", linestyle='-', marker='o')

    # Plot for High Yield (HY) (rating = 1)
    df_hy = df_grouped[df_grouped['rating'] == 1]
    plt.plot(df_hy['date'], df_hy[col], label="HY (High Yield)", linestyle='-', marker='s')

    # Formatting the plot
    plt.xlabel("Date")
    plt.ylabel(col.replace("_", " ").title())  # Format y-label nicely
    plt.title(f"{col.replace('_', ' ').title()} Over Time for IG and HY Bonds")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    




                


