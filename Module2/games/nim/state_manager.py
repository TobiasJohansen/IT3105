from games.nim.utils import next_player

class StateManager():

    def __init__(self, game):
        self.game = game

    def gen_initial_state(self):
        return self.State(self.game.current_player, self.game.remaining_stones)
    
    def gen_child_states(self, state):
        child_states = []
        current_player = next_player[state.current_player]
        for nr_of_selected_stones in range(1, min(state.remaining_stones, self.game.max_selection) + 1):
            remaining_stones = state.remaining_stones - nr_of_selected_stones
            child_states.append((self.State(current_player, remaining_stones), nr_of_selected_stones))
        return child_states

    def is_winning_state(self, state):
        return True if state.n == 0 else False

    class State():
        def __init__(self, current_player, remaining_stones):
            self.current_player = current_player
            self.remaining_stones = remaining_stones