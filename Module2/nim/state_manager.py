from nim.game import Game
import nim.utils

class StateManager():
    def __init__(self, g, p, m, n, k, verbose=False):
        self.g = g
        self.p = p
        self.m = m
        self.n = n
        self.k = k
        self.verbose = verbose
    
    def gen_initial_state(self):
        return Game(self.n, self.k, utils.get_starting_player(self.p), verbose=True)