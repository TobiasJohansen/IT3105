import numpy as np
import random
from anet import ANet
from statemanager import Statemanager
from mct import MCT

statemanager = Statemanager()

def save_anet(g_a, save_interval, anet, path):
    if not g_a % save_interval: 
        anet.save(path + "anet_" + str(g_a))
    
def gen_train_case(d):
    d = d[:]
    state, action_mask = statemanager.gen_case()
    for i, legal_action in enumerate(action_mask):
        if not legal_action:
            d.insert(i, 0)
    return [state, action_mask, d]

def train_models(hex_n, n_actual_games, n_search_games, epsilon, learning_rate, hidden_layer_sizes, 
        hidden_activation_function, optimizer, n_models, path, display_games, g):  
    layer_sizes = [2*(hex_n**2+1)] + hidden_layer_sizes + [hex_n**2]
    anet = ANet(learning_rate, layer_sizes, hidden_activation_function, optimizer)
    save_interval = n_actual_games // (n_models - 1)
    rbuf = []
    for g_a in range(n_actual_games):
        save_anet(g_a, save_interval, anet, path)
        print("Actual game nr:", g_a + 1)
        statemanager.init_state(hex_n)
        mct = MCT(statemanager)
        while not statemanager.final_state():
            for _ in range(n_search_games):
                leaf = mct.find_leaf()
                win = mct.rollout(leaf, anet, epsilon)
                mct.backpropagate(leaf, win)
            children = mct.root.children    
            d = [child.visits for child in children]
            rbuf.append(gen_train_case(d))
            new_root = children[np.argmax(d)]
            mct.update_root(new_root)
            statemanager.state = new_root.state
            if (g_a+1) in display_games:
                statemanager.print_state()
        anet.do_training(rbuf)
    save_anet(n_actual_games, save_interval, anet, path)