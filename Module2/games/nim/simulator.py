from games.nim.game import Game
from games.nim.utils import players
import random

class Simulator():
    
    def __init__(self, game, batch_size, verbose=False):
        self.max_selection = game.max_selection
        self.batch_size = batch_size
        self.verbose = verbose

    def simulate(self, state):
        score = {}
        for player in players:
            score[player] = 0
        nr_of_games = self.batch_size
        if state.is_terminal():
            score[state.winning_player] = nr_of_games
        else:
            for i in range(1, nr_of_games + 1):
                if self.verbose:
                    print("\nGame {0}:".format(i))
                game = Game(state.current_player, state.remaining_stones, self.max_selection, verbose=self.verbose)
                while not game.over:
                    action = random.randint(1, min(self.max_selection, game.remaining_stones))
                    game.select_stones(action)
                score[game.winning_player] += 1
        print("\nBatch Statistics:")
        for player in players:
            wins = score[player]
            percentage = wins / nr_of_games * 100
            print("Player {0} wins {1} out of {2} games ({3:.2f}%).".format(player, wins, nr_of_games, percentage))
        return score

    
        


