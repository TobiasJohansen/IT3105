class State():
    def __init__(self, current_player_idx, remaining_stones):
        self.current_player_idx = current_player_idx
        self.remaining_stones = remaining_stones

    # Used to compare state objects to each other
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__