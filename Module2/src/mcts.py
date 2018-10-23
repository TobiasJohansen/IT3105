import numpy as np
import random

class MCTS():
    def __init__(self, m, rollout_batch_size=1):
        self.m = m
        self.rollout_batch_size = rollout_batch_size
        self.root = None
        self.wins_visits = {}
    
    # Update root to new state
    def update(self, state):
        if self.root and self.root.children:
            for child in self.root.children:
                if child.state == state:
                    self.root = child
                    self.root.parent = None
                    return
        self.root = self.Node(state, None, None)
    
    # Returns the action MCTS decides is the best action in the current root state
    def get_action(self, state_manager):
        for i in range(self.m):
            node = self.tree_search(state_manager)
            leaf = node if state_manager.is_state_terminal(node.state) else self.node_expansion(state_manager, node)
            results = self.leaf_evaluation(state_manager, leaf)
            self.backpropagation(state_manager, leaf, results)
            
            # Used to print tree each m, continuing after user presses enter
            #print("\n{0}/{1}".format(i+1, self.m))
            #self.tree_printer(state_manager, self.root.children, "")
            #input()

        # Find the child node with the highest valued state
        child_scores = []
        children = self.root.children
        for child in children:
            wins, visits =  self.wins_visits[self.get_state_key(state_manager, child)]
            if visits == 0:
                child_scores.append(0)
            else:
                child_scores.append(wins/visits)
        return children[np.argmax(child_scores)].action

    # Returns the unique key of a nodes state
    def get_state_key(self, state_manager, node):
        key = state_manager.get_state_key(node.state)
        if not key in self.wins_visits.keys():
            self.wins_visits[key] = [0, 0]
        return key

    # Returns a non/partially expanded node
    def tree_search(self, state_manager):
        node = self.root
        children = node.children
        while children and all(self.wins_visits[self.get_state_key(state_manager, child)][1] > 0 for child in children):
            node = self.max_UCT(state_manager, node, children)
            children = node.children
        return node
    
    # Calculates UCT scores (exploitation + exploration) and returns the child with the highest score
    def max_UCT(self, state_manager, node, children):
        _, parent_visits = self.wins_visits[self.get_state_key(state_manager, node)]
        child_scores = []
        for child in children:
            wins, visits = self.wins_visits[self.get_state_key(state_manager, child)]
            child_scores.append(wins / visits + np.sqrt(2 * np.log(parent_visits) / visits))
        return children[np.argmax(child_scores)]

    # Returns an unvisited child of a non/partially expanded node 
    def node_expansion(self, state_manager, node):
        children = node.children
        if children:
            unvisited_children = [child for child in children if self.wins_visits[self.get_state_key(state_manager, child)][1] == 0]
            return random.choice(unvisited_children)
        else:
            child_states = state_manager.gen_child_states(node.state)
            for child_state, action in child_states: 
                node.children.append(self.Node(child_state, action, node))
            return random.choice(node.children)

    # Performs rollout on selected node
    def leaf_evaluation(self, state_manager, leaf):
        return state_manager.rollout_state(leaf.state, self.rollout_batch_size)

    # Updates states from leaf to root
    def backpropagation(self, state_manager, node, results):
        while node.parent:
            key = self.get_state_key(state_manager, node)
            self.wins_visits[key][0] += results[node.state.previous_player_idx] 
            self.wins_visits[key][1] += self.rollout_batch_size
            node = node.parent
        self.wins_visits[self.get_state_key(state_manager, node)][1] += self.rollout_batch_size

    # Helper function to print out tree
    def tree_printer(self, state_manager, children, tab):
        if children:
            for child in children:
                print("{0}State: Previous Player Index: {1} - Remaining Stones: {2}"
                    .format(tab, child.state.previous_player_idx, child.state.remaining_stones))
                _, parent_visits = self.wins_visits[self.get_state_key(state_manager, child.parent)]
                wins, visits =  self.wins_visits[self.get_state_key(state_manager, child)]
                q = 0
                uct = np.Infinity
                if visits > 0:
                    q = wins/visits
                    uct = q + np.sqrt(2 * np.log(parent_visits) / visits)
                print("{0}Q(s) - Value: {1}".format(tab, q))
                print("{0}UCT(s) - Value: {1}".format(tab, uct))
                self.tree_printer(state_manager, child.children, tab + str("\t"))

    # Node class used in tree
    class Node():
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []