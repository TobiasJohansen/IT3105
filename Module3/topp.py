import numpy as np
import random
import torch
from statemanager import Statemanager

def play(player_1, player_2, hex_n, display_games, g):
    statemanager = Statemanager()
    players = [player_1, player_2]
    score = [0,0]
    # Play all g games
    for i in range(1,g+1):
        # Display info about game if it is in display_games
        if i in display_games:
            print("{0} vs. {1}: Game {2}:".format(player_1[0], player_2[0], i))
        # Initiate the game state
        statemanager.init_state(hex_n, random.choice([1,2]))
        # Play until game is over
        while not statemanager.final_state():
            # Player is current players corresponding ANET 
            player = players[statemanager.state.player-1]
            # Binarized state and mask is produces to prepare state for being
            # forwarded through ANET
            inputs, mask = statemanager.gen_case()
            # The distribution d is retrieved from ANET
            d = player[1]([inputs],[mask]).detach().numpy().squeeze()
            # The performed action is selected based on the probability distribution d
            board_space = np.random.choice(range(hex_n**2), p=d)
            # The state of statemanager is updated to the new state
            statemanager.state = statemanager.gen_child(board_space)
            # Display state if game is in display_games
            if i in display_games:
                statemanager.print_state()
        score[statemanager.state.winner-1] += 1
    return "{0} vs. {1}: Final score: {2}/{3}".format(player_1[0], player_2[0], score[0], score[1])

def play_tournament(hex_n, n_actual_games, n_search_games, epsilon, learning_rate, hidden_layer_sizes, 
        hidden_activation_function, optimizer, n_models, path, display_games, g):
    # Load trained players
    save_interval = n_actual_games // (n_models - 1)
    players=[]
    for i in range(n_models):
        anet = "anet_" + str(i*save_interval)
        players.append([anet, torch.load(path + anet)])
    # All players play each other in round robin style g times, with a 50% chance at starting each game.
    results = []
    offset = 1
    for player_1 in range(len(players)-1):
        for player_2 in range(offset,len(players)):
            if player_1 != player_2:
                results.append(play(players[player_1], players[player_2], hex_n, display_games, g))
        offset += 1
    
    # Print results
    for result in results:
        print(result)
    
    
