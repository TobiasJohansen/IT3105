from games.nim.simulator import Simulator
from games.nim.state_manager import StateManager
from games.nim.game import Game
from mcts import MCTS

options = {
    "starting_player": 1,
    "total_stones": 10,
    "max_selection": 3
}
game = Game(**options)
mcts = MCTS(10, StateManager(game), Simulator(1, game, verbose=True))

mcts.episode()

