from games.nim.state import State

class Game():

    over = False

    def __init__(self, initial_state, max_selection, nr_of_players, verbose=False):
        self.state = initial_state
        self.max_selection = max_selection
        self.nr_of_players = nr_of_players
        self.players = [i for i in range(1, nr_of_players + 1)]
        self.verbose = verbose

        if self.verbose:
            print("Created a new game of NIM with N = {0} and K = {1}. Player {2} starts!"
                .format(initial_state.remaining_stones, max_selection, initial_state.current_player))

    def select_stones(self, nr_of_stones):     
        self.remove_stones(nr_of_stones)
        self.check_if_won()       
        self.state.current_player = self.next_player(self.state.current_player)
    
    def remove_stones(self, nr_of_stones):
        self.state.remaining_stones -= nr_of_stones
        if self.verbose:
            print("Player {0} selects {1} stone{2}: Remaining stones = {3}"
                .format(self.state.current_player, nr_of_stones, "s" if nr_of_stones > 1 else "", self.state.remaining_stones))
    
    def check_if_won(self):
        if self.state.remaining_stones == 0:
            self.state.winning_player = self.state.current_player
            self.over = True
            if self.verbose:
                print("Player {0} wins!\n".format(self.state.current_player))

    def next_player(self, current_player):
        return self.players[(self.players.index(current_player) + 1) % len(self.players)]
   
    def correct_input(self, nr_of_stones, error_message):
        max_selection = min(self.max_selection, self.state.remaining_stones)
        if not (0 < nr_of_stones <= max_selection):
            print(error_message)
            return False
        return True
    
    def user_input(self):
        answer = False
        error_message = "Select a number in the range [1 - {0}]".format(self.max_selection)
        while not answer:
            answer = input("[Player {0}] - Select stones: ".format(self.state.current_player))
            if answer == "exit":
                exit()
            try:
                answer = int(answer)
            except ValueError:
                answer = False
                print(error_message)
        return self.user_input() if not self.correct_input(answer, error_message) else answer

