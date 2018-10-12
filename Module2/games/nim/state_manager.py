from games.nim.game import Game
import games.nim.utils as utils

class StateManager():

    def __init__(self, starting_player, total_stones, max_selection):
        self.starting_player = starting_player
        self.total_stones = total_stones
        self.max_selection = max_selection

    def gen_initial_state(self):
        return self.State(self.starting_player, self.total_stones)
    
    def gen_child_states(self, state):
        child_states = []
        next_player = utils.next_player[state.current_player]
        for nr_of_selected_stones in range(1, min(state.remaining_stones, self.max_selection) + 1):
            remaining_stones = state.stones - nr_of_selected_stones
            child_states.append(self.State(next_player, remaining_stones))
        return child_states

    def is_winning_state(self, state):
        return True if state.n == 0 else False

    class State():
        def __init__(self, current_player, remaining_stones):
            self.current_player = current_player
            self.remaining_stones = remaining_stones