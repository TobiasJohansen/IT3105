from games.nim.game import Game
from games.nim.utils import players
import random

class Simulator():
    def __init__(self, batch_size, nr_of_simulations, game, verbose=False):
        self.batch_size = batch_size
        self.nr_of_simulations = nr_of_simulations
        self.max_selection = game.max_selection
        self.verbose = verbose

    def simulate(self, state):
        batches = []
        for i in range(self.nr_of_simulations):
            games = []
            if(self.verbose):
                print("\nBatch {0}:".format(i))
            for _ in range(self.batch_size):
                games.append(Game(state.current_player, state.remaining_stones, self.max_selection, verbose=self.verbose))
            batches.append(games)
        wins = 0
        for i, batch in enumerate(batches): 
            if(self.verbose):
                print("\nBatch {0}:".format(i))
            wins += self.rollout(batch)[1]
        return wins / (self.nr_of_simulations * self.batch_size)


    def rollout(self, games):
        score = {}
        for player in players:
            score[player] = 0
        for i, game in enumerate(games):
            if(self.verbose):
                print("\nGame {0}:".format(i))
            while(not game.over):
                action = random.randint(1, min(self.max_selection, game.remaining_stones))
                game.select_stones(action)
            score[game.winning_player] += 1
        if self.verbose:
            wins = score[1]
            total = wins + score[2]
            percentage = wins / total * 100
            print("\nPlayer 1 wins {0} out of {1} games ({2:.2f}%).".format(wins, total, percentage))
        return score

    
        


