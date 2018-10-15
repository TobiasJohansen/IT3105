from games.nim.simulator import Simulator
from games.nim.state_manager import StateManager
from games.nim.game import Game
from mcts import MCTS

options = {
    "starting_player": 1,
    "total_stones": 10,
    "max_selection": 3,
    "verbose": True
}

print("\n=== NIM - MTCS ===\n")
game = Game(**options)
mcts = MCTS(4, StateManager(game), Simulator(game, 2, True))

while not game.over:
    game.select_stones(mcts.episode())
    if not game.over:
        game.select_stones(int(input("Select stones: ")))

