from games.nim.state_manager import StateManager
from games.nim.game import Game
from src.mcts import MCTS

print("\n=== NIM - MTCS ===\n")

nr_of_mcts_players = 1
total_players = 2

game = Game(starting_player=2, nr_of_players=total_players, total_stones=10, max_selection=3)

mcts_players = {}
for i in range(1 + total_players - nr_of_mcts_players, total_players + 1):
    mcts_players[i] = MCTS(player_number=i, m=100, batch_size=100, state_manager=StateManager(game=game), verbosity_level=0)

while not game.over:
    current_player = game.current_player
    if current_player in mcts_players.keys():
        game.select_stones(mcts_players[current_player].episode())
    else:
        input_given = False
        while not input_given:
            input_given = game.select_stones(game.user_input())