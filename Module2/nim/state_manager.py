from nim.game import Game
from nim.state import State
import nim.utils as utils

class StateManager():

    def __init__(self, p, n, k):
        self.p = p
        self.n = n
        self.k = k

    def gen_initial_state(self):
        return State(self.p, self.n)
    
    def gen_child_states(self, state):
        child_states = []
        n = state.n
        for k in range(1, min(n, self.k) + 1):
            next_player = 1 if state.p == 2 else 2
            next_n = n - k
            child_states.append(State(next_player, next_n))
        return child_states