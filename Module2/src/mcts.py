import numpy as np
import random

class MCTS():
    def __init__(self, player_number, m, batch_size, state_manager):
        self.player_number = player_number
        self.m = m
        self.batch_size = batch_size
        self.state_manager = state_manager
            
    def episode(self):
        self.root = self.Node(self.state_manager.gen_initial_state(), None, None)
        self.root.children = []  
        for _ in range(self.m):
            node = self.tree_search()
            leaf = node if self.state_manager.is_state_terminal(node.state) else self.node_expansion(node)
            wins = self.leaf_evaluation(leaf)
            self.backpropagation(leaf, wins)
            #print("\nM {0}/{1}".format(i, self.m))
            #print("\nRollout results: {0}".format(wins))
        #self.tree_printer(self.root, "")
        #print("\nBest action is:", self.root.children[np.argmax([child.wins / child.visits for child in self.root.children])].action)
        #exit()
            #input()
        return self.root.children[np.argmax([child.wins / child.visits for child in self.root.children])].action

    def tree_printer(self, node, tab):
        for child in node.children:
            print("\n{0}Action: {1} ==> {2}"
                .format(tab, child.action, self.state_manager.to_string(child.state)))
            print("{0}{1}/{2}".format(tab, child.wins, child.visits))
            #if child.children:
            #    self.tree_printer(child, tab+str("\t"))

    # Returns a non/partially expanded node
    def tree_search(self):
        node = self.root
        children = node.children
        while children and all(child.visits > 0 for child in children):
            if node.state.current_player == self.player_number:
                node = self.max_UCT(children)
            else:
                node = self.min_UCT(children)
            children = node.children
        return node
    
    def max_UCT(self, children):
        return children[np.argmax([child.wins / child.visits + np.sqrt(2) * np.sqrt(np.log(child.parent.visits) / child.visits) for child in children])]

    def min_UCT(self, children):
        return children[np.argmin([child.wins / child.visits - np.sqrt(2) * np.sqrt(np.log(child.parent.visits) / child.visits) for child in children])]

    # Returns an unvisited child of a non/partially expanded node 
    def node_expansion(self, node):
        children = node.children
        if children:
            unvisited_children = [child for child in children if child.visits == 0]
            return random.choice(unvisited_children)
        else:
            child_states = self.state_manager.gen_child_states(node.state)
            for child_state, action in child_states: 
                node.children.append(self.Node(child_state, action, node))
            return random.choice(node.children)

    def leaf_evaluation(self, leaf):
        return self.state_manager.rollout_state(leaf.state, self.batch_size)

    def backpropagation(self, node, wins):
        parent = node.parent
        while parent:
            previous_player = parent.state.current_player
            player_wins = wins[previous_player]
            perspective_score = player_wins if previous_player == self.player_number else - player_wins 
            node.wins += perspective_score
            node.visits += self.batch_size
            node = parent
            parent = parent.parent
        node.visits += self.batch_size

    class Node():
        
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0