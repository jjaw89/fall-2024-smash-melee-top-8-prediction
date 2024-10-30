# Because some computations can be SLOW, we can process them in split files.
# The file names should be name_temp_<num>.pkl

import datetime
from glicko2 import Player
import multiprocessing
import numpy as np
import os
import pandas as pd
import re
import time
from tqdm.auto import tqdm

#tqdm.pandas()

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
    file_prefix = input("Enter file name without _temp_NUM.pkl: ")

    # Find all files of the proper form
    all_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    matching_files = []
    for f in all_files:
        if not f.startswith(file_prefix):
            continue

        file_suffix = f.removeprefix(file_prefix)

        if re.fullmatch(r'\_temp\_\d+.pkl', file_suffix):
            matching_files.append(f)

    return file_prefix, matching_files

# For sub-processes. Actually compute the elos for the corresponding file.
def process_file(file_prefix, file, position, total_processes):
    grouped_df = pd.read_pickle(data_path + file)
    
    initial_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2024, 12, 31) # TODO: Properly compute this instead of just guessing
                                               #       Though this might be harder with coordinating multiple files.
    interval = datetime.timedelta(weeks=1)

    unique_players = list(grouped_df['player_char_char'].unique())

    # TODO: Surely there's a more professional way to do this bit.
    dates = {0: initial_date}

    date = initial_date + interval
    i = 1

    while date <= end_date:
        dates[i] = date

        date += interval
        i += 1

    # Convenient store of glicko objects
    glicko_objects = {}
    for player in unique_players:
        glicko_objects[player] = Player()

    # Pre-allocating the dataframe for maximum efficiency.
    player_ratings_df = pd.DataFrame([[1500.0] * len(unique_players)], columns=unique_players, index=list(dates.values()))

    # The actual code to compute elo for each player/char/char combo.
    # This will be done by a groupby and apply
    def compute_pcc_elo(x):
        # player/char/char
        pcc = x.iloc[0]['pcc_duplicate']

        # More easily allow for getting the week number
        x = x.set_index('end_index')

        glicko_object = glicko_objects[pcc]

        # More efficient to keep track of where every occuring week number is (as an iloc).
        weeknum_to_iloc = [-1]*len(dates)
        for i in range(0, len(x.index)):
            weeknum_to_iloc[x.index[i]] = i

        for index in dates:
            if weeknum_to_iloc[index] == -1:
                glicko_object.did_not_compete()
            else:
                glicko_object.update_player(x.iloc[weeknum_to_iloc[index]]['opponent_rating'],
                                            x.iloc[weeknum_to_iloc[index]]['opponent_rd'],
                                            x.iloc[weeknum_to_iloc[index]]['outcome'])

            # Bugfix stuff
            MIN_ELO = 500.0
            MAX_RD = 350.0
            if glicko_object.getRating() < MIN_ELO:
                glicko_object.setRating(MIN_ELO)

            if glicko_object.getRd() > MAX_RD:
                glicko_object.setRd(MAX_RD)

            player_ratings_df.loc[initial_date + index*interval, pcc] = glicko_object.getRating()

    mininterval = 1.0
    tqdm.pandas(position=position, mininterval=mininterval, desc='File ' + str(position))

    # If the updates of multiple progress bars are too close to one another, the output is messed up
    time.sleep(position * mininterval / total_processes)

    grouped_df.groupby('player_char_char').progress_apply(compute_pcc_elo, include_groups=False)

    player_ratings_df.to_pickle(data_path + file_prefix + '_processed_' + str(file_number) + '.pkl')

if __name__ == '__main__':
    processes = []

    file_prefix, matching_files = get_matching_files()

    # Start separate processes
    for file in matching_files:
        # Slightly janky way of getting the file number
        file_suffix = file.removeprefix(file_prefix)
        file_number = int(re.sub(r'[^\d]', '', file_suffix))

        p = multiprocessing.Process(target=process_file, args=(file_prefix, file, file_number, len(matching_files)))
        p.start()
        processes.append(p)
    
    # Wait for all of them to finish
    for p in processes:
        p.join()

    # Done
    print("Done!")