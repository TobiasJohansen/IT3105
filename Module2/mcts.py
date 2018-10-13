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
        for _ in range(self.m):
            node = self.tree_search()
            leaf = random.choice(self.node_expansion(node))
            print(self.leaf_evaluation(leaf))

    def tree_search(self):
        node = self.root
        children = node.children
        while children and all(child.visited > 0 for child in children):
            node = children[np.argmax([child.q + child.u for child in children])]
            children = node.children
        
        if not children:
            return node
        
        unvisited_children = []
        for child in children:
            if child.visited == 0:
                unvisited_children.append(child)
        return random.choice(unvisited_children)

    def node_expansion(self, node):
        child_states = self.state_manager.gen_child_states(node.state)
        for child_state, action in child_states: node.children.append(self.Node(child_state, action, node))
        return node.children

    def leaf_evaluation(self, leaf):
        return self.simulator.simulate(leaf.state)

    class Node():
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []
            self.q = 0
            self.u = 0
            self.visited = 0