import copy
from games.nim.game import Game
from games.nim.state import State
import numpy as np
import random

class StateManager():

    def __init__(self, game):
        self.game = game
    
    # States are unique solely based on remaining stones
    def get_state_key(self, state):
        return state.remaining_stones
    
    def is_state_terminal(self, state):
        return not state.remaining_stones

    # Generates all possible child states of specified state
    def gen_child_states(self, state):
        child_states = []
        current_player_idx = self.game.next_player_idx(state.current_player_idx)
        for nr_of_selected_stones in range(1, min(state.remaining_stones, self.game.max_selection) + 1):
            remaining_stones = state.remaining_stones - nr_of_selected_stones
            child_state = State(current_player_idx, remaining_stones)
            child_states.append((child_state, nr_of_selected_stones))
        return child_states

    # Plays out a specified number of games from a specified state by making random moves from both players side
    def rollout_state(self, parent_state, state, rollout_batch_size):
        results = np.zeros(len(self.game.players), dtype=int)
        if self.is_state_terminal(state):
            results[parent_state.current_player_idx] = rollout_batch_size
        else:
            for _ in range(rollout_batch_size):
                game = Game(self.game.players, state.current_player_idx, state.remaining_stones, self.game.max_selection, verbose=False)
                while not game.over:
                    nr_of_stones = random.randint(1, min(self.game.max_selection, game.state.remaining_stones))
                    game.select_stones(nr_of_stones)
                results[game.winning_player_idx] += 1
        return results
