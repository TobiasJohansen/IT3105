from games.nim.statemanager import StateManager
from games.nim.game import Game
from src.mcts import MCTS
import random

print("\n=== NIM - MTCS ===\n")

nr_of_mcts_players = 2
total_players = 2
nr_of_games = 1000

scores = {}
for player in range(1, total_players + 1):
    scores[player] = 0

for i in range(1, nr_of_games + 1):
    print("Game {0} / {1}".format(i, nr_of_games))
    starting_player = random.choice([1,2])
    game = Game(starting_player=starting_player, nr_of_players=total_players, total_stones=10, max_selection=3, verbosity_level=0)
    mcts_players = {}
    ##for i in range(1 + total_players - nr_of_mcts_players, total_players + 1):
    mcts_players[1] = MCTS(player_number=1, m=50, nr_of_rollout_games=2, state_manager=StateManager(game=game), verbosity_level=0)
    mcts_players[2] = MCTS(player_number=2, m=100, nr_of_rollout_games=1, state_manager=StateManager(game=game), verbosity_level=0)
    while not game.over: 
        current_player = game.current_player
        if current_player in mcts_players.keys():
            game.select_stones(mcts_players[current_player].episode())
        else:
            input_given = False
            while not input_given:
                input_given = game.select_stones(game.user_input())
    scores[game.winning_player] += 1
print(scores)