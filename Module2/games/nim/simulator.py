from games.nim.game import Game
from games.nim.state import State
from games.nim.statemanager import StateManager
from games.player import Player
import numpy as np
import random
from src.mcts import MCTS

class Simulator():
    def __init__(self, player_names):
        self.player_names = player_names
    
    def simulate_batch(self, g, p, m, n, k, rollout_batch_size=1, verbose=False):
        # Tracking score in batch
        score = np.zeros(len(self.player_names), dtype=int)
        print("\nSimulating games:\n")
        for i in range(1, g + 1):
            print("Game {0}/{1}".format(i, g))
            # The shared brain between both bots
            brain = MCTS(m, rollout_batch_size)
            starting_player_idx = random.choice([idx for idx in range(len(self.player_names))]) if p == "Mix" else p
            # Creating the main game
            game = Game([Player(player_name, brain) for player_name in self.player_names], starting_player_idx, n, k, verbose)
            # And a state manager to help the brain understand the game
            state_manager = StateManager(game)
            # While the game is still active, update the brain to the current game.state and perform the next action
            while not game.over:
                brain.update(game.state)
                game.select_stones(game.current_player().take_turn(game, state_manager))
            # Increment the score of the winning player
            score[game.winning_player_idx] += 1
        return [(self.player_names[i], score[i]) for i in range(len(score))]
            