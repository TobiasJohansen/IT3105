from games.nim.game import Game
from games.nim.state import State
from games.nim.statemanager import StateManager
from src.mcts import MCTS

class Simulator():
    def __init__(self, nr_players):
        self.nr_of_players = nr_players

    def simulate_batch(self, g, p, m, n, k, batch_size=1, verbose=False):
        score = {}
        for player in range(1, self.nr_of_players + 1):
            score[player] = 0
        print()
        for i in range(1, g + 1):
            print("Game {0}/{1}".format(i, g))
            game = Game(State(p, n), k, self.nr_of_players, verbose=verbose)
            players = {}
            for i in range(1, self.nr_of_players + 1):
                players[i] = MCTS(i, m, batch_size, StateManager(game=game))
            while not game.over:
                current_player = game.state.current_player
                nr_of_stones = players[current_player].episode()
                game.select_stones(nr_of_stones)
            score[game.state.winning_player] += 1
        return score
            