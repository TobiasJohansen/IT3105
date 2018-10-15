import numpy as np
import random 

class MCTS():
    def __init__(self, player_number, m, batch_size, state_manager, verbosity_level=1):
        self.player_number = player_number
        self.m = m
        self.batch_size = batch_size
        self.state_manager = state_manager
        self.state_manager.verbosity_level = verbosity_level
        self.verbosity_level = verbosity_level
            
    def episode(self):
        self.root = self.Node(self.state_manager.gen_initial_state(), None, None)
        self.root.children = []
        for i in range(1, self.m + 1):
            if self.verbosity_level > 0:
                print("\nM - {0}:".format(i))
            node = self.tree_search()
            leaf = node if node.state.is_terminal() else self.node_expansion(node)
            score = self.leaf_evaluation(leaf)
            self.backpropagation(leaf, score)
        if self.verbosity_level > 0:
            print()
        return self.root.children[np.argmax([child.q for child in self.root.children])].action

    # Returns a non/partially expanded node
    def tree_search(self):
        node = self.root
        children = node.children
        while children and all(child.visits > 0 for child in children):
            if node.state.current_player == self.player_number:
                node = children[np.argmax([child.q + child.u for child in children])]
            else:
                node = children[np.argmin([child.q - child.u for child in children])]
            children = node.children
        return node

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
        if self.verbosity_level > 0:
            print("\nEvaluation of node:")
            print(leaf.to_string())
        return self.state_manager.simulate_state(leaf.state, self.batch_size, self.verbosity_level)

    def backpropagation(self, leaf, score):
        games_won = score[self.player_number]
        games_played = self.batch_size
        leaf.wins += games_won
        leaf.visits += games_played  
        node = leaf
        parent = leaf.parent
        while parent:     
            parent.wins += games_won
            parent.visits += games_played      
            node.q = node.wins / node.visits 
            node.u = 1 * np.sqrt(np.log(parent.visits) / node.visits)
            node = parent
            parent = parent.parent     

    class Node():
        
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.q = 0
            self.u = 0
            self.children = []
            self.wins = 0
            self.visits = 0
        
        def to_string(self):
            depth = 0
            parent = self.parent
            while parent:
                depth += 1
                parent = parent.parent
            return "Tree depth: {0}\nPrevious action: {1}\n".format(depth, self.action) + self.state.to_string()