# fall-2024-smash-melee-top-8-prediction

The data for this project comes from https://github.com/smashdata/ThePlayerDatabase . It will be /data/melee_player_database.db and we ignore it in the .gitignore file.

To Do:
-make an enviroment.yaml for conda that matches the docker container. We might need to ignore the packages that depend on Cuda.
-Fix the glicko-2 matchup rating calculation so that keep track of the number of games of that matchup a player has played up to that date.
-Set up a feature (funcion that searches for) the PvP match history between the players playing the set.
