class State():
    def __init__(self, previous_player_idx, remaining_stones):
        self.previous_player_idx = previous_player_idx
        self.remaining_stones = remaining_stones

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__