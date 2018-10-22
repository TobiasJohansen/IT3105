from games.player import Player
from games.nim.game import Game
from games.nim.statemanager import StateManager

from src.mcts import MCTS

brain = MCTS(m=1000, batch_size=100)
players = [Player(name="Per", brain=brain), Player(name="Pål", brain=brain)]
game = Game(players=players, starting_player_idx=0, total_nr_of_stones=15, max_selection=3, verbose=True)
state_manager = StateManager(game)

while not game.over:
    brain.update(game.state)
    game.select_stones(game.current_player().take_turn(game, state_manager))
    


