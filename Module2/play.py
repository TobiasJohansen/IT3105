from games.nim.statemanager import StateManager
from games.nim.game import Game
from games.nim.state import State
from src.mcts import MCTS

print("\n=== NIM - MTCS ===\n")

nr_of_mcts_players = 1
total_players = 2

while True:
    game = Game(initial_state=State(current_player=1, remaining_stones=10), max_selection=3, nr_of_players=total_players)
    mcts_players = {}
    for i in range(1 + total_players - nr_of_mcts_players, total_players + 1):
        mcts_players[i] = MCTS(player_number=i, m=50, batch_size=1, state_manager=StateManager(game=game), verbosity_level=0)
    while not game.over:
        current_player = game.state.current_player
        if current_player in mcts_players.keys():
            game.select_stones(mcts_players[current_player].episode())
        else:
            game.select_stones(game.user_input())
    print()