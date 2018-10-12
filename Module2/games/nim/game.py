import itertools

class Game():

    over = False

    def __init__(self, n, k, turns, verbose=False):
               
        self.n = n 
        self.k = k
        self.verbose = verbose 

        player_manager = itertools.cycle(turns)
        self.next_player = lambda : next(player_manager)
        self.current_player = self.next_player()

        if self.verbose:
            print("Created a new game of NIM with N = {0} and K = {1}. Player {2} starts!"
                .format(self.n, self.k, self.current_player))
    
    def select_stones(self, n):        
        if self.is_active() and self.correct_input(n):
            self.remove_stones(n)
            self.was_game_won()          
            self.current_player = self.next_player()
            return True
        return False
        
    def is_active(self):
        if self.over:
            if self.verbose:
                print("Game has ended, start a new one!")
            return False
        return True
        
    def correct_input(self, n):
        max_grab = min(self.k, self.n)
        if not (0 < n <= max_grab):
            if self.verbose:
                print("Select a number in the range [1 - {0}]".format(max_grab))
            return False
        return True
    
    def remove_stones(self, n):
        self.n -= n
        if self.verbose:
            print("Player {0} selects {1} stone{2}: Remaining stones = {3}"
                .format(self.current_player, n, "s" if n > 1 else "", self.n))
    
    def was_game_won(self):
        if self.n == 0:
            self.winning_player = self.current_player
            self.over = True
            if self.verbose:
                print("Player {0} wins!".format(self.current_player))
        
