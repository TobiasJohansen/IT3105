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
        self.check_input(g, p, m, n, k, rollout_batch_size, verbose)

        # Arrays to keep count of number of games started / won
        starting_player_idx_count = np.zeros(len(self.player_names), dtype=int) 
        results = np.zeros(len(self.player_names), dtype=int)
        
        print("\nSimulating games:\n")
        for i in range(1, g + 1):
            print("Game {0}/{1}".format(i, g))
            
            # Increase count of current starting player
            starting_player_idx = p
            if starting_player_idx == "Mix":
                starting_player_idx = random.choice([idx for idx in range(len(self.player_names))])
            starting_player_idx_count[starting_player_idx] += 1

            # Create the brain of the MCTS AI(s), set up the Game and initialize the StateManager
            brain = MCTS(m, rollout_batch_size)
            game = Game([Player(player_name, brain) for player_name in self.player_names], starting_player_idx, n, k, verbose)
            state_manager = StateManager(game)
            
            # Play until game is over
            while not game.over:
                # Update the internal state of the AI's brain
                brain.update(game.state)
                # Request action from current player and execute it
                game.select_stones(game.current_player().take_turn(game, state_manager))

            # Update results
            results[game.winning_player_idx] += 1

        # Print results
        print("\nGames started:")
        for i in range(len(starting_player_idx_count)):
            print("[P{0} - {1}] started {2} of {3} games.".format(i, self.player_names[i], starting_player_idx_count[i], g))
        print("\nGames won:")
        for i in range(len(results)):
            print("[P{0} - {1}] won {2} of {3} games. ({4}%)".format(i, self.player_names[i], results[i], g, 100.0*results[i]/g))  
    
    def check_input(self, g, p, m, n, k, rollout_batch_size=1, verbose=False):
        if not (type(g) == int and 1 <= g):
            raise ValueError("G should be an integer equal to or larger than 1")
        if not (type(p) == int and p in range(len(self.player_names)) or (type(p) == str and p == "Mix")):
            raise ValueError("P should be an integer in the range 0 to {0} or the string \"Mix\"".format(len(self.player_names) - 1))
        if not (type(m) == int and 1 <= m):
            raise ValueError("M should be an integer equal to or larger than 1")
        if not (type(n) == int and k <= n):
            raise ValueError("N should be an integer equal to or larger than {0}".format(k))
        if not (type(k) == int and 1 <= k <= n):
            raise ValueError("K should be an integer in the range 1 to {0}".format(n))
        if not (type(rollout_batch_size) == int and 1 <= rollout_batch_size):
            raise ValueError("Rollout Batch Size should be an integer equal to or larger than 1")
        if not (type(verbose) == bool):
            raise ValueError("Verbose should be a boolean")