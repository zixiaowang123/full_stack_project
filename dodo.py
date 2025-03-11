"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys
import json

sys.path.insert(1, "./src/")

import shutil
from os import environ, getcwd, path
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import ctypes

from colorama import Fore, Style, init

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter

from settings import config

try:
    in_slurm = environ["SLURM_JOB_ID"] is not None
except:
    in_slurm = False


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        doit_mark = stuff.split(" ")[0].ljust(2)
        task = " ".join(stuff.split(" ")[1:]).strip() + "\n"
        output = (
            Fore.GREEN
            + doit_mark
            + f" {path.basename(getcwd())}: "
            + task
            + Style.RESET_ALL
        )
        self.outstream.write(output)


if not in_slurm:
    DOIT_CONFIG = {
        "reporter": GreenReporter,
        # other config here...
        # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
        "backend": "sqlite3",
        "dep_file": "./.doit-db.sqlite",
    }
else:
    DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
init(autoreset=True)


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
PUBLISH_DIR = config("PUBLISH_DIR")
#USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --log-level WARN --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --log-level WARN --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --log-level WARN --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import ctypes

from settings import config


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
        kycrspid: identification id
        kytreasno: identification id
        mcaldt: date of report
        tmpubout: face value of public outstanding
        treas_yld: annualized discount rate
        tmatdt: maturity date
    '''
    issue_df = issue_df.dropna(subset=['kycrspid', 'kytreasno'])
    tm_df = tm_df.dropna(subset=['kycrspid', 'kytreasno'])
    
    # merge in maturity date
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
        yield, -- Yield to maturity at report date
        cusip,
        isin,
        rating, -- Combined Rating Class: 0.IG or 1.HY
        conv, -- Flag Convertible (1 or 0)
        offering_price, -- Offering Price
        price_eom, -- Price-End of Month (reported price)
        t_spread -- Avg Bid/Ask Spread
    
    treas_df: DF of treasuries with columns
        kycrspid, kytreasno: identification for treasuries
        mcaldt: date when the data was collected (Monthly Date)
        tmpubout: face value of bonds of this type held by the public
        treas_yld: yield of the treasury (annualized)
        tmatdt: date when the treasury matures

    day_window: window to find successful matching for treasuries

    output: dataframe with treasury data merged into all valid ones for bond_df
        company_symbol, -- Company Symbol (issuer stock ticker).
        maturity, -- Date that the issue's principal is due for repayment.
        amount_outstanding, -- The amount of the issue remaining outstanding.
        date, -- Monthly Date (report date)
        yield, -- ytm at report date
        cusip,
        rating, -- Combined Rating Class: 0.IG or 1.HY
        price_eom, -- Price-End of Month (reported price)
        t_spread, -- Avg Bid/Ask Spread
        treas_yld, -- yield of similar treasury
    '''

    # assume all necessary filtering has been done on bond df

    # pre work for datetime functionality and filtering
    treas_df = treas_df.reset_index()
    bond_df = bond_df.reset_index()

    bond_df = bond_df[['cusip', 'company_symbol', 'date', 'maturity', 
                   'amount_outstanding', 'yield', 'rating', 'price_eom', 't_spread']]

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
    maturity_dates = pd.Series(bond_df['maturity'].unique())
    valid_mats = maturity_dates.apply(
        lambda x: any(mat_date - timedelta(days=day_window) <= x <= mat_date + timedelta(days=day_window) for mat_date in treas_mat_set)
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
        t_df = t_df[t_df['day_diff'] <= day_window]

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

    # date, and maturity are definitely not missing, need all of the following values for categorization
    # insurance drop NaN
    merge_df = merge_df.dropna(subset=['cusip', 'v_id', 'yield', 'treas_yld', 'rating'])
    desired_cols = ['cusip', 'company_symbol', 'date', 'maturity', 'amount_outstanding', 'yield', 'rating', 'price_eom', 't_spread', 'treas_yld']
    
    merge_df = merge_df[desired_cols]

    return merge_df


def merge_red_code_into_bond_treas(bond_treas_df, red_c_df):
    '''
    bond_treas_df: dataframe containing the merged bond and treasury data
        company_symbol, -- Company Symbol (issuer stock ticker).
        maturity, -- Date that the issue's principal is due for repayment.
        amount_outstanding, -- The amount of the issue remaining outstanding (sum of face values).
        date, -- Monthly Date (report date)
        yield, -- ytm at report date
        cusip,
        rating, -- Combined Rating Class: 0.IG or 1.HY
        price_eom, -- Price-End of Month (reported price)
        t_spread, -- Avg Bid/Ask Spread
        treas_yld, -- yield of similar treasury
    red_c_df: dataframe containing red code merging information
        redcode, -- redcode of the issuer
        ticker, -- ticker of the issuer
        obl_cusip, -- cusip of an issue, the first 6 objects characters of the string should be the issuers tag 
        isin, -- these are product specific
        tier -- tier of product


    output: dataframe with the issuer cusip and red_code now added
        company_symbol, -- Company Symbol (issuer stock ticker).
        maturity, -- Date that the issue's principal is due for repayment.
        amount_outstanding, -- The amount of the issue remaining outstanding (sum of face values).
        date, -- Monthly Date (report date)
        yield, -- ytm at report date
        issuer_cusip, -- 6 character issuer cusip
        rating, -- Combined Rating Class: 0.IG or 1.HY
        price_eom, -- Price-End of Month (reported price)
        t_spread, -- Avg Bid/Ask Spread
        treas_yld, -- yield of similar treasury
        redcode -- redcode is issuer specific, used to merge CDS values later on
    '''

    bond_treas_df['issuer_cusip'] = bond_treas_df.apply(lambda row: row['cusip'][:6], axis=1)

    red_c_df = red_c_df[['obl_cusip', 'redcode']].dropna()
    red_c_df['issuer_cusip'] = red_c_df.apply(lambda row: row['obl_cusip'][:6], axis=1)
    
    # only need these 2 to merge
    red_c_df = red_c_df[['issuer_cusip', 'redcode']].drop_duplicates().reset_index(drop=True)

    # should drop all uneeded elements
    merged_df = bond_treas_df.merge(red_c_df, on='issuer_cusip', how='inner')

    return merged_df

##################################
## Begin rest of PyDoit tasks here
##################################

def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }

def task_data_pull_1():
    file_dep = [
        "./src/settings.py",
        "./src/pull_markit_mapping.py",
        "./src/pull_wrds_bonds.py",
        "./src/pull_treasury_rates.py",
    ]
    targets = [
        DATA_DIR / "RED_and_ISIN_mapping.parquet",
        DATA_DIR / "wrds_bond.parquet",
        DATA_DIR / "monthly_ts_data.parquet",
        DATA_DIR / "issue_data.parquet.parquet",
    ]

    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_markit_mapping.py",
            "ipython ./src/pull_wrds_bonds.py",
            "ipython ./src/pull_treasury_rates.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    } 

TREASURY_ISSUE_FILE_NAME = "issue_data.parquet"
TREASURY_MONTHLY_FILE_NAME = "monthly_ts_data.parquet"
CORPORATES_MONTHLY_FILE_NAME = "wrds_bond.parquet"
RED_CODE_FILE_NAME = "RED_and_ISIN_mapping.parquet"

def task_create_bond_treas_redcode_file():
    """
    Main function to load data, process it, and merge Treasury data into Bonds.
    """
    print("Loading data...")

    # Load DataFrames
    issue_df = pd.read_parquet(f"{DATA_DIR}/{TREASURY_ISSUE_FILE_NAME}")
    treas_monthly_df = pd.read_parquet(f"{DATA_DIR}/{TREASURY_MONTHLY_FILE_NAME}")
    bond_df = pd.read_parquet(f"{DATA_DIR}/{CORPORATES_MONTHLY_FILE_NAME}")
    red_df = pd.read_parquet(f"{DATA_DIR}/{RED_CODE_FILE_NAME}")

    print("Generating Treasury data...")
    treasury_data = generate_treasury_data(issue_df, treas_monthly_df)
    
    print("Merging Treasuries into Bonds...")
    bond_treas_df = merge_treasuries_into_bonds(bond_df, treasury_data, day_window=3)

    print("Merging Redcodes into file...")
    bond_red_df = merge_red_code_into_bond_treas(bond_treas_df, red_df)

    print("Saving processed data...")
    bond_red_df.to_parquet(f"{DATA_DIR}/merged_bond_treasuries_redcode.parquet")

    print("Processing complete. Data saved.")

def task_generate_redcode_dict():
    data = pd.read_parquet(f"{DATA_DIR}/merged_bond_treasuries_redcode.parquet")
    data['year'] = data["date"].dt.year

    red_code_dict = {}
    for y in list(data['year'].unique()):
        y_df = data[data['year'] == y]
        red_code_dict[y] = list(y_df['redcode'].unique())
    d = {int(key): value for key, value in red_code_dict.items()}
    with open(f"{DATA_DIR}/red_code_dict.json", 'w') as file:
        json.dump(d, file)

#DATA PULL TAKES FOREVER, UNCOMMENT IF NEEDED
'''def task_data_pull_2():
    """ """
    file_dep = [
        "./_data/red_code_dict.json"
        "./src/settings.py",
        "./src/pull_mergent_bonds.py",
        "./src/pull_markit_cds_1.py",
        "./src/pull_markit_cds_2.py",
    ]
    targets = [
        DATA_DIR / "mergent_bond.parquet",
        DATA_DIR / "markit_cds_1.parquet",
        DATA_DIR / "markit_cds_2.parquet",
    ]

    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_treasury_rates.py",
            "ipython ./src/pull_markit_cds_1.py",
            "ipython ./src/pull_markit_cds_2.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }'''

'''
notebook_tasks = {
    "01_example_notebook_interactive.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "02_example_with_dependencies.ipynb": {
        "file_dep": ["./src/pull_fred.py"],
        "targets": [Path(OUTPUT_DIR) / "GDP_graph.png"],
    },
    "03_public_repo_summary_charts.ipynb": {
        "file_dep": [
            "./src/pull_fred.py",
            "./src/pull_ofr_api_data.py",
            "./src/pull_public_repo_data.py",
        ],
        "targets": [
            OUTPUT_DIR / "repo_rate_spikes_and_relative_reserves_levels.png",
            OUTPUT_DIR / "rates_relative_to_midpoint.png",
        ],
    },
}


def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./src") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_file(
                    Path("./src") / f"{notebook_name}.ipynb",
                    OUTPUT_DIR / f"{notebook_name}.ipynb",
                    mkdir=True,
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                OUTPUT_DIR / f"{notebook_name}.ipynb",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }'''
# fmt: on


# ###############################################################
# ## Task below is for LaTeX compilation
# ###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        # "./reports/report_example.tex",
        # "./reports/my_article_header.sty",
        # "./reports/slides_example.tex",
        # "./reports/my_beamer_header.sty",
        # "./reports/my_common_header.sty",
        # "./reports/report_simple_example.tex",
        # "./reports/slides_simple_example.tex",
        # "./src/example_plot.py",
        # "./src/example_table.py",
        "./reports/final_report.tex",

    ]
    targets = [
        "./reports/final_report.pdf",
        # "./reports/report_example.pdf",
        # "./reports/slides_example.pdf",
        # "./reports/report_simple_example.pdf",
        # "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/final_report.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/final_report.tex",  # Clean
            # "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            # "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            # "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            # "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # # Simple templates based on small adjustments to Overleaf templates
            # "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            # "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            # "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            # "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
            #
            # Example of compiling and cleaning in another directory. This often fails, so I don't use it
            # f"latexmk -xelatex -halt-on-error -cd -output-directory=../_output/ ./reports/report_example.tex",  # Compile
            # f"latexmk -xelatex -halt-on-error -c -cd -output-directory=../_output/ ./reports/report_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }
'''
notebook_sphinx_pages = [
    "./docs/notebooks/EX_" + notebook.split(".")[0] + ".html"
    for notebook in notebook_tasks.keys()
]
sphinx_targets = [
    "./docs/index.html",
    "./docs/myst_markdown_demos.html",
    "./docs/apidocs/index.html",
    *notebook_sphinx_pages,
]

def task_compile_sphinx_docs():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        OUTPUT_DIR / ("_" + notebook.split(".")[0] + ".py")
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./pipeline.json",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook generate -f",
        ],  # Use docs as build destination
        # "actions": ["sphinx-build -M html ./docs/ ./docs/_build"], # Previous standard organization
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": ["run_notebooks",],
        "clean": True,
    }'''


###############################################################
## Uncomment the task below if you have R installed. See README
###############################################################


# def task_install_r_packages():
#     """Example R plots"""
#     file_dep = [
#         "r_requirements.txt",
#         "./src/install_packages.R",
#     ]
#     targets = [OUTPUT_DIR / "R_packages_installed.txt"]

#     return {
#         "actions": [
#             "Rscript ./src/install_packages.R",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


# def task_example_r_script():
#     """Example R plots"""
#     file_dep = [
#         "./src/pull_fred.py",
#         "./src/example_r_plot.R"
#     ]
#     targets = [
#         OUTPUT_DIR / "example_r_plot.png",
#     ]

#     return {
#         "actions": [
#             "Rscript ./src/example_r_plot.R",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "task_dep": ["pull_fred"],
#         "clean": True,
#     }


# rmarkdown_tasks = {
#     "04_example_regressions.Rmd": {
#         "file_dep": ["./src/pull_fred.py"],
#         "targets": [],
#     },
#     # "04_example_regressions.Rmd": {
#     #     "file_dep": ["./src/pull_fred.py"],
#     #     "targets": [],
#     # },
# }


# def task_knit_RMarkdown_files():
#     """Preps the RMarkdown files for presentation format.
#     This will knit the RMarkdown files for easier sharing of results.
#     """
#     # def knit_string(file):
#     #     return f"""Rscript -e "library(rmarkdown); rmarkdown::render('./src/04_example_regressions.Rmd', output_format='html_document', output_dir='./_output/')"""
#     str_output_dir = str(OUTPUT_DIR).replace("\\", "/")
#     def knit_string(file):
#         """
#         Properly escapes the quotes and concatenates so that this will run.
#         The single line version above was harder to get right because of weird
#         quotation escaping errors.

#         Example command:
#         Rscript -e "library(rmarkdown); rmarkdown::render('./src/04_example_regressions.Rmd', output_format='html_document', output_dir='./_output/')
#         """
#         return (
#             "Rscript -e "
#             '"library(rmarkdown); '
#             f"rmarkdown::render('./src/{file}.Rmd', "
#             "output_format='html_document', "
#             f"output_dir='{str_output_dir}')\""
#         )

#     for notebook in rmarkdown_tasks.keys():
#         notebook_name = notebook.split(".")[0]
#         file_dep = [f"./src/{notebook}", *rmarkdown_tasks[notebook]["file_dep"]]
#         html_file = f"{notebook_name}.html"
#         targets = [f"{OUTPUT_DIR / html_file}", *rmarkdown_tasks[notebook]["targets"]]
#         actions = [
#             # "module use -a /opt/aws_opt/Modulefiles",
#             # "module load R/4.2.2",
#             knit_string(notebook_name)
#         ]

#         yield {
#             "name": notebook,
#             "actions": actions,
#             "file_dep": file_dep,
#             "targets": targets,
#             "clean": True,
#             # "verbosity": 1,
#         }


###################################################################
## Uncomment the task below if you have Stata installed. See README
###################################################################

# if OS_TYPE == "windows":
#     STATA_COMMAND = f"{config.STATA_EXE} /e"
# elif OS_TYPE == "nix":
#     STATA_COMMAND = f"{config.STATA_EXE} -b"
# else:
#     raise ValueError(f"OS_TYPE {OS_TYPE} is unknown")

# def task_example_stata_script():
#     """Example Stata plots

#     Make sure to run
#     ```
#     net install doenv, from(https://github.com/vikjam/doenv/raw/master/) replace
#     ```
#     first to install the doenv package: https://github.com/vikjam/doenv.
#     """
#     file_dep = [
#         "./src/pull_fred.py",
#         "./src/example_stata_plot.do",
#     ]
#     targets = [
#         OUTPUT_DIR / "example_stata_plot.png",
#     ]
#     return {
#         "actions": [
#             f"{STATA_COMMAND} do ./src/example_stata_plot.do",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "task_dep": ["pull_fred"],
#         "clean": True,
#         "verbosity": 2,
#     }
