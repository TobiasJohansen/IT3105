class Game():

    over = False

    def __init__(self, starting_player, nr_of_players, total_stones, max_selection, verbosity_level=2):
        self.current_player = starting_player
        self.nr_of_players = nr_of_players
        self.players = [i for i in range(1, nr_of_players + 1)]
        self.remaining_stones = total_stones
        self.max_selection = max_selection
        self.verbosity_level = verbosity_level
        
        if self.verbosity_level > 1:
            print("Created a new game of NIM with N = {0} and K = {1}. Player {2} starts!"
                .format(total_stones, max_selection, starting_player))
    
    def select_stones(self, nr_of_stones):        
        if self.is_active() and self.correct_input(nr_of_stones):
            self.remove_stones(nr_of_stones)
            self.check_if_won()          
            self.current_player = self.next_player(self.current_player)
            return True
        return False
        
    def is_active(self):
        if self.over:
            if self.verbosity_level > 1:
                print("Game has ended, start a new one!")
            return False
        return True
        
    def correct_input(self, nr_of_stones):
        max_selection = min(self.max_selection, self.remaining_stones)
        if not (0 < nr_of_stones <= max_selection):
            print("Select a number in the range [1 - {0}]".format(max_selection))
            return False
        return True
    
    def remove_stones(self, nr_of_stones):
        self.remaining_stones -= nr_of_stones
        if self.verbosity_level > 1:
            print("Player {0} selects {1} stone{2}: Remaining stones = {3}"
                .format(self.current_player, nr_of_stones, "s" if nr_of_stones > 1 else "", self.remaining_stones))
    
    def check_if_won(self):
        if self.remaining_stones == 0:
            self.winning_player = self.current_player
            self.over = True
            if self.verbosity_level > 1:
                print("Player {0} wins!".format(self.current_player))

    def next_player(self, current_player):
        return self.players[(self.players.index(current_player) + 1) % len(self.players)]
   
    def user_input(self):
        answer = False
        while not answer:
            answer = input("[Player {0}] - Select stones: ".format(self.current_player))
            if answer == "exit":
                exit()
            try:
                answer = int(answer)
            except ValueError:
                answer = False
                print("Select a number in the range [1 - {0}]".format(self.max_selection))
        return self.user_input() if not self.correct_input(answer) else answer