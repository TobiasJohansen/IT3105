import numpy as np
import random 

class MCTS():
    def __init__(self, m, state_manager, simulator):
        self.m = m
        self.state_manager = state_manager
        self.simulator = simulator
            
    def episode(self):
        self.root = self.Node(self.state_manager.gen_initial_state(), None, None)
        self.root.children = []
        for i in range(1, self.m + 1):
            print("\nM - {0}:".format(i))
            node = self.tree_search()
            leaf = node if node.state.is_terminal() else self.node_expansion(node)
            score = self.leaf_evaluation(leaf)
            self.backpropagation(leaf, score)
        print()
        return self.root.children[np.argmax([child.q for child in self.root.children])].action

    # Returns a non/partially expanded node
    def tree_search(self):
        node = self.root
        children = node.children
        while children and all(child.visits > 0 for child in children):
            if node.state.current_player == 1:
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
        return self.simulator.simulate(leaf.state)

    def backpropagation(self, leaf, score):
        games_won = score[1]
        games_played = 0
        for _, wins in score.items():
            games_played += wins
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