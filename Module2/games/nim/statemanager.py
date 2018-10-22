import copy
from games.nim.game import Game
from games.nim.state import State
import random

class StateManager():

    def __init__(self, game):
        self.game = game

    def gen_initial_state(self):
        return copy.deepcopy(self.game.state)
    
    def is_state_terminal(self, state):
        return not state.remaining_stones

    def gen_child_states(self, state):
        child_states = []
        current_player = self.game.next_player(state.current_player)
        for nr_of_selected_stones in range(1, min(state.remaining_stones, self.game.max_selection) + 1):
            remaining_stones = state.remaining_stones - nr_of_selected_stones
            winning_player = state.current_player if remaining_stones == 0 else None
            child_state = State(current_player, remaining_stones, winning_player)
            child_states.append((child_state, nr_of_selected_stones))
        return child_states

    def rollout_state(self, state, batch_size):
        wins = {}
        for player in self.game.players:
            wins[player] = 0
        if state.winning_player:
            wins[state.winning_player] = batch_size
        else:
            for _ in range(batch_size):
                game = Game(copy.deepcopy(state), self.game.max_selection, self.game.nr_of_players)
                while not game.over:
                    nr_of_stones = random.randint(1, min(self.game.max_selection, game.state.remaining_stones))
                    game.select_stones(nr_of_stones)
                wins[game.state.winning_player] += 1
        return wins
    
    def to_string(self, state):
        return "Current Player: {0} - Remaining Stones: {1}".format(state.current_player, state.remaining_stones) 

    
