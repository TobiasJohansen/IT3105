import numpy as np
import random 

class MCTS():
    def __init__(self, state_manager, simulator):
        self.state_manager = state_manager
        self.simulator = simulator
        self.root = self.Node(self.state_manager.gen_initial_state(), None, None)
        self.root.children = []
            
    def episode(self):
        self.root = self.Node(self.state_manager.gen_initial_state(), None, None)
        self.root.children = [] 
        node = self.tree_search()
        leaf = random.choice(self.node_expansion(node))
        self.leaf_evaluation(leaf)

    def tree_search(self):
        node = self.root
        children = node.children
        while(children):
            node = children[np.argmax([child.q + child.u for child in children])]
            children = node.children
        return node

    def node_expansion(self, node):
        child_states = self.state_manager.gen_child_states(node.state)
        for child_state, action in child_states: node.children.append(self.Node(child_state, action, node))
        return node.children

    def leaf_evaluation(self, leaf):
        print(self.simulator.simulate(leaf.state))

    class Node():
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []
            self.q = 0
            self.u = 0
            self.visited = 0