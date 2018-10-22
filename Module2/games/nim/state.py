class State():
        
        def __init__(self, current_player, remaining_stones, winning_player=None):
            self.current_player = current_player
            self.remaining_stones = remaining_stones
            self.winning_player = winning_player