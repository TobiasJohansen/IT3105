from nim.game import Game
import utils

class Simulator():
    def __init__(self, g, p, m, n, k, verbose=False):
        
        self.g = g
        self.p = p
        self.m = m
        self.n = n
        self.k = k
        self.verbose = verbose

        games = []
        for _ in range(g):
            games.append(Game(n, k, utils.get_starting_player(self.p), True))


