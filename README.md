# Super Smash Bros Melee prediction

Predicts the outcome of both individual sets, and the final winner of the top 8, out of Super Smash Bros Melee tournaments.


## About

This project was the capstone project of the Erdos Institute's [Data Science Boot Camp](https://www.erdosinstitute.org/programs/fall-2024/data-science-boot-camp). It was written over approximately a three month period by Dan Ursu and Jaspar Wiart.

Super Smash Bros Melee is a fighting game released for the Nintendo GameCube in 2001. While several newer installments in the franchise have been released since then, Melee continues to regularly be played in tournaments offering tens of thousands of dollars in prize money, and pull in hundreds of thousands of viewers online.

Our aim in this capstone project is the same as for any other sport - predict the winner. In particular, we had two main goals in mind:
* Be able to accurately predict the outcome of individual sets between two players.
* Given the final eight players (top 8) in a tournament, predict who the final winner of the tournament will be.

There already exists a very widely used ELO-based rating system called the [Glicko rating system](https://en.wikipedia.org/wiki/Glicko_rating_system), and we wanted to see if we could perform *better* than a baseline model of "whoever has the higher ELO". In many instances, especially for predicting the outcomes of individual sets, the answer is a definitive *yes*.

In many ways, this project was heavily focused on feature engineering and processing large amounts of data efficiently, rather than blindly tossing some super sophisticated AI model at the data and calling it a day.


## Methodology and results

### Single-set models

Data on previous Super Smash Bros tournaments (as SQLite databases) was obtained from [this GitHub repo](https://github.com/smashdata/ThePlayerDatabase). After some initial data cleanup, we computed weekly ratings of each of the players, and this establishes our baseline for single-set data (whoever has the higher ELO). A slightly more sophisticated model is also included to account for various factors (such as ELO being exactly the default starting value of 1500.0 and being inaccurate).

The following showcases how well our baseline models perform on data from 2024. The sample sizes involved are enormous (tens to hundreds of thousands), and we obtain quite tight 95% confidence intervals.

| **Baseline models / Accuracy**     | All sets      | Top 8 sets    |
|------------------------------------|---------------|---------------|
| "Whoever has the higher ELO"       | 77.56 +- 0.16 | 73.89 +- 0.36 |
| XGBoost trained only on player ELO | 79.05 +- 0.16 | 74.04 +- 0.36 |

From here, our main goal was to improve upon the above baselines by engineering extra features that a machine learning model may use. The following highlights our main successful attempts at doing so.

First and foremost, Super Smash Bros is a fighting game where each player initially chooses a specific character to fight with, each with their own unique moveset. It is therefore extremely plausible that different players (who themselves usually tend to favor only one or two characters) will perform better or worse against different characters. We thought of different ways to account for this, and ultimately settled on the following:

* For any combination of (player, character_1, character_2), compute an adjusted ELO score that only takes into account when player was playing as character_1, and their opponent was playing as character_2. We created two slightly different implementations for accomplishing this that we nicknamed "alt" and "alt2", but we eventually ended up only using "alt2".

* Treat unique combinations of (player, character) as completely separate players. Then compute another resulting set of ELOs, nicknamed "alt3".

Moreover, the following are easy extra features to toss in:

* Based on how the Glicko-2 algorithm works, at any point in time we have not just an ELO score, but also a "ratings deviation" (RD) that acts as a sort of "measurement error" on their true skill level. This applies both to the default ELOs and engineered ones above.

* Simply counting how many times someone's ELO score was updated up to a certain point, as an alternate rough estimate to the above RD values.

Finally, we also have:

* Keeping track of how often any pair of players (player_1, player_2) have played before, and tossing the results of their last ten matches in as a list of features.

We again trained an XGBoost model on the above list of features, and we obtain a small but noticeable (and evidently statistically significant) increase in accuracy.

| **Models / Accuracy**              | All sets      | Top 8 sets    |
|------------------------------------|---------------|---------------|
| "Whoever has the higher ELO"       | 77.56 +- 0.16 | 73.89 +- 0.36 |
| XGBoost trained only on player ELO | 79.05 +- 0.16 | 74.04 +- 0.36 |
| XGBoost on all engineered features | 79.89 +- 0.16 | 75.03 +- 0.35 |



### Top 8 models

With the above single-set prediction model in hand (which is capable of outputting probabilities of winning, and not just binary predictions), we moved on to predicting 



## Technical details about running the code

There is a slight bias in terms of player 1 winning a set over player 2 (approximately 70%), and it is unclear if this is due to player seeding, or just due to the data being manually entered later. To account for this, we randomize player 1 and player 2.

A python implementation of the Glicko-2 algorithm (ELO-like rating) was found [here](https://github.com/deepy/glicko2). In some sense, Glicko2 ratings are time-series data for each of the players. In another sense, they are not, as each of the ratings already takes into account all of their past performance to compute their skill level up to a certain point. Nevertheless, to avoid any possibility of data leakage, we always trained our models on data up to a certain point, and tested on data after.