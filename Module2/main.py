from games.nim.state_manager import StateManager
from mcts import MCTS

mcts = MCTS(StateManager(1, 10, 3))