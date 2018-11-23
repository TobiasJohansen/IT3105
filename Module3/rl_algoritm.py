import numpy as np
import random
from anet import ANet
from statemanager import Statemanager
from mct import MCT

statemanager = Statemanager()

# Makes the model save itself
def save_anet(g_a, save_interval, anet, path):
    if not g_a % save_interval: 
        anet.save(path + "anet_" + str(g_a))

# Creates a training case for ANET based on a state and distribution of visits d
def gen_train_case(d):
    d = d[:]
    state, action_mask = statemanager.gen_case()
    for i, legal_action in enumerate(action_mask):
        if not legal_action:
            d.insert(i, 0)
    d = np.array(d)
    return [state, action_mask, (d / sum(d)).tolist()]

# Trains n_models models
def train_models(hex_n, n_actual_games, n_search_games, epsilon, learning_rate, hidden_layer_sizes, 
        hidden_activation_function, optimizer, n_models, path, display_games, g):  
    # Configure ANET
    layer_sizes = [2*(hex_n**2+1)] + hidden_layer_sizes + [hex_n**2]
    anet = ANet(learning_rate, layer_sizes, hidden_activation_function, optimizer)
    save_interval = n_actual_games // (n_models - 1)
    rbuf = []
    # Play actual games
    for g_a in range(n_actual_games):
        # Consider saving
        save_anet(g_a, save_interval, anet, path)
        # Start a new Hex game
        print("Actual game nr:", g_a + 1)
        statemanager.init_state(hex_n)
        # Create the MCT
        mct = MCT(statemanager)
        # Play until game is over
        while not statemanager.final_state():
            # Simulate n_search_games games
            for _ in range(n_search_games):
                leaf = mct.find_leaf()
                win = mct.rollout(leaf, anet, epsilon)
                mct.backpropagate(leaf, win)
            # Select the action corresponding to the child with the most visits
            children = mct.root.children  
            # Create a case for ANET to train on  
            d = [child.visits for child in children]
            rbuf.append(gen_train_case(d))
            # Update states of MCT and statemanager
            new_root = children[np.argmax(d)]
            mct.update_root(new_root)
            statemanager.state = new_root.state
            # Display state if game is in display_games
            if (g_a+1) in display_games:
                statemanager.print_state()
        # Train ANET
        anet.do_training(rbuf)
    # Consider saving one more time before ending
    save_anet(n_actual_games, save_interval, anet, path)