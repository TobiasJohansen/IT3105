from games.nim.game import Game
import random

class StateManager():

    def __init__(self, game):
        self.game = game

    def gen_initial_state(self):
        return self.State(self.game.current_player, self.game.remaining_stones)
    
    def gen_child_states(self, state):
        child_states = []
        current_player = self.game.next_player(state.current_player)
        for nr_of_selected_stones in range(1, min(state.remaining_stones, self.game.max_selection) + 1):
            remaining_stones = state.remaining_stones - nr_of_selected_stones
            child_state = self.State(current_player, remaining_stones)
            child_state.winning_player = state.current_player if child_state.is_terminal() else None
            child_states.append((child_state, nr_of_selected_stones))
        return child_states

    def simulate_state(self, state, batch_size, verbosity_level):
        score = {}
        for player in self.game.players:
            score[player] = 0
        if state.is_terminal():
            score[state.winning_player] = batch_size
        else:
            for i in range(1, batch_size + 1):
                if verbosity_level > 1:
                    print("\nGame {0}:".format(i))
                game = Game(state.current_player, self.game.nr_of_players, state.remaining_stones, self.game.max_selection, verbosity_level=verbosity_level)
                while not game.over:
                    action = random.randint(1, min(self.game.max_selection, game.remaining_stones))
                    game.select_stones(action)
                score[game.winning_player] += 1
        if verbosity_level > 0:
            print("\nBatch Statistics:")
            for player in self.game.players:
                wins = score[player]
                percentage = wins / batch_size * 100
                print("Player {0} wins {1} out of {2} games ({3:.2f}%)".format(player, wins, batch_size, percentage))
        return score

    class State():
        
        def __init__(self, current_player, remaining_stones):
            self.current_player = current_player
            self.remaining_stones = remaining_stones

        def is_terminal(self):
            return not self.remaining_stones

        def to_string(self):
            return "Current Player: {0} \nRemaining Stones: {1}".format(self.current_player, self.remaining_stones)
