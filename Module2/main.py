from games.nim.simulator import Simulator
from games.nim.state_manager import StateManager
from games.nim.game import Game
from mcts import MCTS

game = Game(1, 5, 3)
state_manager = StateManager(game)
simulator=Simulator(100, 100, game)
mcts = MCTS(state_manager, simulator)
mcts.episode()

