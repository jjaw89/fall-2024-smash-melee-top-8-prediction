# Super Smash Bros Melee prediction

Predicts the outcome of both individual sets, and the final winner of the top 8, out of Super Smash Bros Melee tournaments.


## About

This project was the capstone project of the Erdos Institute's [Data Science Boot Camp](https://www.erdosinstitute.org/programs/fall-2024/data-science-boot-camp). It was written over approximately a three month period by Dan Ursu and Jaspar Wiart.

Super Smash Bros Melee is a fighting game released for the Nintendo GameCube in 2001. While several newer installments in the franchise have been released since then, Melee continues to regularly be played in tournaments offering tens of thousands of dollars in prize money, and pull in hundreds of thousands of viewers online.

Our aim in this capstone project is the same as for any other sport - predict the winner. In particular, we had two main goals in mind:
* Be able to accurately predict the outcome of individual sets between two players.
* Given the final eight players (top 8) in a tournament, predict who the final winner of the tournament will be.

There already exists a very widely used ELO-based rating system called the [Glicko rating system](https://en.wikipedia.org/wiki/Glicko_rating_system), and we wanted to see if we could perform *better* than a baseline model of "whoever has the higher ELO". In many instances, especially for predicting the outcomes of individual sets, the answer is a definitive *yes*.

In many ways, this project was heavily focused on feature engineering and processing large amounts of data efficiently (millions of rows of data to work with), rather than blindly tossing some super sophisticated AI model at the data and calling it a day.


## Methodology and results

### Single-set models

Data on previous Super Smash Bros tournaments (as SQLite databases) was obtained from [this GitHub repo](https://github.com/smashdata/ThePlayerDatabase). After some initial data cleanup, we computed weekly ratings of each of the players, and this establishes our baseline for single-set data (whoever has the higher ELO).

From here, our main goal was to improve upon the above baselines by engineering extra features that a machine learning model may use. The following highlights our main successful attempts at doing so.

First and foremost, Super Smash Bros is a fighting game where each player initially chooses a specific character to fight with, each with their own unique moveset. It is therefore extremely plausible that different players (who themselves usually tend to favor only one or two characters) will perform better or worse against different characters. We thought of different ways to account for this, and ultimately settled on the following:

* For any combination of (player, character_1, character_2), compute an adjusted ELO score that only takes into account when player was playing as character_1, and their opponent was playing as character_2. We created two slightly different implementations for accomplishing this that we nicknamed "alt" and "alt2", but we eventually ended up only using "alt2".

* Treat unique combinations of (player, character) as completely separate players. Then compute another resulting set of ELOs, nicknamed "alt3".

Moreover, the following are easy extra features to toss in:

* Based on how the Glicko-2 algorithm works, at any point in time we have not just an ELO score, but also a "ratings deviation" (RD) that acts as a sort of "measurement error" on their true skill level. This applies both to the default ELOs and engineered ones above.

* Simply counting how many times someone's ELO score was updated up to a certain point, as an alternate rough estimate to the above RD values.

Finally, we also have:

* Keeping track of how often any pair of players (player_1, player_2) have played before, and tossing the results of their last ten matches in as a list of features.

We again trained an XGBoost model on the above list of features, and we obtain a small but noticeable (and evidently statistically significant) increase in accuracy. All models are finally tested on data from 2024, and the number of data points in the test set is quite large (in the hundreds of thousands), hence the fairly small 95% confidence intervals.

| **Models / Accuracy**              | All sets      | Top 8 sets    |
|------------------------------------|---------------|---------------|
| "Whoever has the higher ELO"       | 77.56 +- 0.16 | 73.89 +- 0.36 |
| XGBoost trained only on player ELO | 79.05 +- 0.16 | 74.04 +- 0.36 |
| XGBoost on all engineered features | 79.89 +- 0.16 | 75.03 +- 0.35 |



### Top 8 models

With the above single-set prediction model in hand (which is capable of outputting probabilities of winning, and not just binary predictions), we moved on to predicting the winner of the final eight players (top 8) out of any given tournament.

We will mention this ahead of time, but unfortunately the results obtained here are not nearly as definitive as in the single-set case. Specifically, there is a fairly good couple of baseline models that have unusually good performance.

* The first is simply choosing the player with the highest ELO out of the top 8.

* The second comes from the observation that, based on how most typical top 8 bracket structures work, it is substantially more likely that a player from the four winners' side brackets wins. Hence, we could also just choose the person with the highest elo in the winners' semifinals (WSF).

For figuring out additional features that we could use - we have data on what sets a player has played to get to the top 8, and we moreover have our single-set predictor from before that can also output reasonably accurate win/loss probabilities, not just the binary outcome. As such, the two main features that come to mind are:

* Computing how "likely" it was that they made it to the top 8, given the probabilities of winning/losing each set that they took to get there. The intention is to see if they are having an unusually good or bad performance during the specific tournament that they are playing.

* Computing the pairwise probabilities that each of the top 8 players have for winning/losing against each other.

In short, while these additional features seem to hold their own, the difference is not significant enough to tell if they have a similar performance or not. This is tested on data from 2024, with about 5000 data points.

| **Models**                                  | **Accuracy** |
|---------------------------------------------|--------------|
| "Whoever has the higher ELO out of top 8"   | 67.6 +- 1.3  |
| "Whoever has the higher ELO out of WSF (4)" | 70.2 +- 1.3  |
| XGBoost on engineered features              | 70.0 +- 1.3  |
| XGBoost on engineered features + ELO        | 70.1 +- 1.3  |

A small note is that the single-set predictor was trained on data up to the end of 2022, and then the top 8 predictor was trained on data from 2023 only, in order to avoid data leakage.



## Technical details about running the code

The temporary workspace for manipulating data will be the ``data`` folder. To start, the initial database containing past tournament info should be downloaded from [this repo](https://github.com/smashdata/ThePlayerDatabase), and ``melee_player_database.db`` should be placed in the ``data`` folder. This folder will serve as our main workspace for generating temporary/necessary files.

### Preprocessing

Afterwards, run the notebooks in the ``preprocessing`` folder in order. These extract the tables from the above database, and perform some basic preprocessing on them, including:

* Randomizing player 1 and player 2, as there is a noticeable bias (around 70%) for player 1 winning. It is unclear if this is due to player seeding or from manually entering the winner.

* Extracting tournaments (and in particular majors as a separate file as well), labeling the top 8 players according to the brackets they are in (winners' side, losers' side, ...), and listing the sets that they have played in order to get to the top 8. It also extracts information from sets that have data on the individual games played (including what characters each player played as).

### Feature engineering (Glicko-2 rating / ELO scores)

Now turn to the ``feature_engineering`` folder, and run each of the files. This will generate Glicko-2 ratings (ELO scores) for each of the players, along with three different variations that take into account which characters they have been playing. The library we are using to compute these scores is [this one](https://github.com/deepy/glicko2).

Note that ``default_elo.ipynb`` and ``engineered_elo_alt3.ipynb`` can be run standalone. For each of ``engineered_elo_alt.ipynb`` and ``engineered_elo_alt2.ipynb``, the following extra steps need to be taken:

* Run ``engineered_elo_alt.ipynb`` and ``engineered_elo_alt2.ipynb`` to generate split data files that will be processed afterwards in parallel. This will generate a bunch of ``(...)_temp_N.pkl`` files.

* Run ``compute_elos_for_splits.py`` and follow the instructions on how to process these files (this uses multiprocessing and speeds up the resulting computation substantially). It will output a bunch of ``(...)_processed_N.pkl`` files.

* Run ``remerge.py`` and follow the instructions on merging the above files back into one.

### Generating the final dataset

Open the ``dataset_generation`` folder, and run the ``dataset_generator.ipynb`` file. It will take all of the necessary information from whatever was generated in the ``data`` folder, and combine it into one final ``


In some sense, Glicko2 ratings are time-series data for each of the players. In another sense, they are not, as each of the ratings already takes into account all of their past performance to compute their skill level up to a certain point. Nevertheless, to avoid any possibility of data leakage, we always trained our models on data up to a certain point, and tested on data after.