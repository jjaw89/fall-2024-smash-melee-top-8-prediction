# Super Smash Bros Melee prediction

Predicts the outcome of both individual sets, and the final winner of the top 8, out of Super Smash Bros Melee tournaments.


## About

This project was the capstone project of the Erdos Institute's [Data Science Boot Camp](https://www.erdosinstitute.org/programs/fall-2024/data-science-boot-camp). It was written over approximately a three month period by Dan Ursu and Jaspar Wiart.

Super Smash Bros Melee is a fighting game released for the Nintendo GameCube in 2001. While several newer installments in the franchise have been released since then, Melee continues to regularly be played in tournaments offering tens of thousands of dollars in prize money, and pull in hundreds of thousands of viewers online.

Our aim in this capstone project is the same as for any other sport - predict the winner. In particular, we had two main goals in mind:
* Be able to accurately predict the outcome of individual sets between two players.
* Given the final eight players (top 8) in a tournament, predict who the final winner of the tournament will be.

There already exists a very widely used ELO-based rating system called the [Glicko rating system](https://en.wikipedia.org/wiki/Glicko_rating_system), and we wanted to see if we could perform *better* than a baseline model of "whoever has the higher ELO". In many instances, especially for predicting the outcomes of individual sets, the answer is yes.


## Methodology

Data on previous Super Smash Bros tournaments (as SQLite databases) was obtained from [this GitHub repo](https://github.com/smashdata/ThePlayerDatabase).