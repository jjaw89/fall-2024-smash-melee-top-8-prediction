import numpy as np
import os
import pandas as pd
import re

if os.path.exists('/workspace/data'):
    # Load the dictionary of DataFrames from the pickle
    data_path = '/workspace/data/'
else:
    data_path = '../data/'

# For some reason, visual studio code sets the current working directory for this file to the main repo folder
# and NOT to the folder that this file is contained in. This ensures a consistent working directory.
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Gets the file prefix that we are looking for,
# along with all of the corresponding files
def get_matching_files():
    # file_prefix = 'char_vs_char_player_rankings_weekly_alt2'
    file_prefix = input("Enter file name without _processed_NUM.pkl: ")

    # Find all files of the proper form
    all_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    matching_files = []
    for f in all_files:
        if not f.startswith(file_prefix):
            continue

        file_suffix = f.removeprefix(file_prefix)

        if re.fullmatch(r'\_processed\_\d+.pkl', file_suffix):
            matching_files.append(f)

    return file_prefix, matching_files


file_prefix, matching_files = get_matching_files()
dataframes = []

for file in matching_files:
    df = pd.read_pickle(data_path + file)
    dataframes.append(df)

# Rows should be indexed by date, and should just match up.
overall_df = pd.concat(dataframes, axis=1)
overall_df.to_pickle(data_path + file_prefix + '.pkl')