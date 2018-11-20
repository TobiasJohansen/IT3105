import numpy as np
import random

class MCTS():
    def __init__(self, m, rollout_batch_size=1):
        self.m = m
        self.rollout_batch_size = rollout_batch_size
        self.root = None
            
    # When game state changes, update root of network to its child if it exists
    def update(self, state):
        if self.root and self.root.children:
            for child in self.root.children:
                if child.state == state:
                    self.root = child
                    self.root.parent = None
                    return
        self.root = self.Node(state, None, None)
    
    # Get best action for current state
    def get_action(self, state_manager):
        for _ in range(self.m):
            # Gets non/partially-expanded node
            node = self.tree_search()
            # Expands and returns a child node it if it is non-expanded, returns an unvisited child-node if it is expanded.
            # Does nothing if state is terminal 
            leaf = node if state_manager.is_state_terminal(node.state) else self.node_expansion(state_manager, node)
            # Evaluates the node, returning a score of the games played from that node on the form [p1_wins, p2_wins, etc.]
            wins = self.leaf_evaluation(state_manager, leaf)
            # Backpropagate the nodes from leaf to root and update visits for all nodes
            # Wins are updated by the number of wins the player moving into it had in the simulations 
            self.backpropagation(leaf, wins)
        # Returns the best action in roots children
        return self.root.children[np.argmax([0 if child.visits == 0 else child.wins / child.visits for child in self.root.children])].action

    def printer(self, children, tab):
        if children:
            for child in children:
                print(tab,child.state.remaining_stones,child.state.previous_player_idx)
                self.printer(child.children, tab + str("\t"))

    # Returns a non/partially expanded node
    def tree_search(self):
        node = self.root
        children = node.children
        while children and all(child.visits > 0 for child in children):
            node = self.max_UCT(children)
            children = node.children
        return node
    
    def max_UCT(self, children):
        return children[np.argmax([child.wins / child.visits + np.sqrt(2 * np.log(child.parent.visits) / child.visits) for child in children])]

    # Returns an unvisited child of a non/partially expanded node 
    def node_expansion(self, state_manager, node):
        children = node.children
        if children:
            unvisited_children = [child for child in children if child.visits == 0]
            return random.choice(unvisited_children)
        else:
            child_states = state_manager.gen_child_states(node.state)
            for child_state, action in child_states: 
                node.children.append(self.Node(child_state, action, node))
            return random.choice(node.children)

    def leaf_evaluation(self, state_manager, leaf):
        return state_manager.rollout_state(leaf.state, self.rollout_batch_size)

    def backpropagation(self, node, wins):
        while node.parent:
            node.wins += wins[node.state.previous_player_idx] 
            node.visits += self.rollout_batch_size
            node = node.parent
        node.visits += self.rollout_batch_size

    class Node():
        
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0