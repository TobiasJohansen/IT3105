from games.nim.state import State

class Game():

    over = False

    def __init__(self, players, starting_player_idx, total_nr_of_stones, max_selection, verbose=True):        
        
        self.players = players
        self.current_player_idx = starting_player_idx
        self.state = State(None, total_nr_of_stones)
        self.max_selection = max_selection
        self.verbose = verbose
        
        if self.verbose:
            print("\nCreated a new game of NIM with N = {0} and K = {1}. [P{2} - {3}] starts!"
                .format(total_nr_of_stones, max_selection, starting_player_idx + 1, players[starting_player_idx].name))
    
    # Returns the current player as a player object  
    def current_player(self):
        return self.players[self.current_player_idx]
    
    # Removes stones, checks if game is won and updates current/previous player
    def select_stones(self, nr_of_stones):     
        self.remove_stones(nr_of_stones)
        self.check_if_won()  
        self.state.previous_player_idx = self.current_player_idx
        self.current_player_idx = self.next_player_idx(self.current_player_idx)
    
    def remove_stones(self, nr_of_stones):
        self.state.remaining_stones -= nr_of_stones
        if self.verbose:
            print("[P{0} - {1}] selects {2} stone{3}: Remaining stones = {4}"
                .format(self.current_player_idx + 1, self.players[self.current_player_idx].name, nr_of_stones,
                    "s" if nr_of_stones > 1 else "", self.state.remaining_stones))
    
    def check_if_won(self):
        if self.state.remaining_stones == 0:
            self.winning_player_idx = self.current_player_idx
            self.over = True
            if self.verbose:
                print("[P{0} - {1}] wins!\n".format(self.current_player_idx + 1, self.players[self.current_player_idx].name))

    def next_player_idx(self, current_player):
        if current_player == None:
            return self.current_player_idx
        return (current_player + 1) % len(self.players)

    # Checks if input given is correct or not
    def correct_input(self, inpt):
        try:
            inpt = int(inpt)
            max_selection = min(self.max_selection, self.state.remaining_stones)
            if not (0 < inpt <= max_selection):
                raise ValueError
            return inpt
        except ValueError:
            print("Select a number in the range [1 - {0}]".format(self.max_selection))
            return None
    
    # Prompts user for input while given input is incorrect
    def user_input(self):
        inpt = input("[P{0} - {1}] Select stones: ".format(self.current_player_idx + 1, self.players[self.current_player_idx].name))
        if inpt == "exit" : exit()
        correct_inpt = self.correct_input(inpt)
        return correct_inpt if correct_inpt else self.user_input()

