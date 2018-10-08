import random

turns_options = { 
    1: [1, 2], 
    2: [2, 1]
}

def get_starting_player(p):
    if p == "Mix":
        return turns_options[random.randint(1,2)]
    return turns_options[p]
        
