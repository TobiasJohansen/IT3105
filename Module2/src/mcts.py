import numpy as np
import random

class MCTS():
    def __init__(self, m, rollout_batch_size=1):
        self.m = m
        self.rollout_batch_size = rollout_batch_size
        self.root = None
        self.wins_visits = {}
    
    # Update root to new state
    def update(self, state_manager, state):
        # If new state is in one of roots nodes, update root to that child
        if self.root and self.root.children:
            for child in self.root.children:
                if child.state == state:
                    self.root = child
                    self.root.parent = None
        else:
            self.root = self.Node(state, None, None)
        # If new root has no children, generate them
        if not self.root.children:
            self.node_expansion(state_manager, self.root)
    
    # Returns the action MCTS decides is the best action in the current root state
    def get_action(self, state_manager):
        for i in range(self.m):  
            node = self.root
            # Repeat if all states of the added child nodes has been visited before
            while node.children and all(self.get_visits(state_manager, child.state) > 0 for child in node.children):
                node = self.tree_search(state_manager, node)
                # Expand if state of node has been visited before
                if self.get_visits(state_manager, node.state) > 0:
                    self.node_expansion(state_manager, node)   
            # Perform one last tree search to select node with an unvisited state
            node = self.tree_search(state_manager, node)         
            # Evaluate that state
            results = self.leaf_evaluation(state_manager, node)
            # Update states
            self.backpropagation(state_manager, node, results)

            # Used to print tree each m, continuing after user presses enter
            # print("\n{0}/{1}".format(i+1, self.m))
            # self.tree_printer(state_manager, self.root.children, "", 2)
            # input()
        
        # Find the child node with the highest valued state
        return self.get_UCT(state_manager, self.root, self.root.children, explore=False).action

    # Returns a non/partially expanded node
    def tree_search(self, state_manager, node):
        children = node.children
        while children:
            node = self.get_UCT(state_manager, node, children)
            children = node.children
        return node

    # Returns an unvisited child of a non/partially expanded node 
    def node_expansion(self, state_manager, node):
        child_states = state_manager.gen_child_states(node.state)
        for child_state, action in child_states: 
            node.children.append(self.Node(child_state, action, node))

    # Performs rollout on selected node
    def leaf_evaluation(self, state_manager, node):
        return state_manager.rollout_state(node.parent.state, node.state, self.rollout_batch_size)

    # Updates states from leaf to root
    def backpropagation(self, state_manager, node, results):
        while node:
            if node.parent:
                winning_player_idx = node.parent.state.current_player_idx 
                self.increase_wins(state_manager, node.state, results[winning_player_idx]) 
            self.increase_visits(state_manager, node.state, self.rollout_batch_size)
            node = node.parent
    
    # Calculates UCT scores (exploitation + exploration) and returns the child with the highest score
    def get_UCT(self, state_manager, node, children, explore=True):
        _, parent_visits = self.get_wins_visits(state_manager, node.state)
        child_scores = []
        for child in children:
            wins, visits = self.get_wins_visits(state_manager, child.state)
            exploitation = 0
            exploration = np.Infinity
            if visits:
                exploitation = wins / visits
                exploration = np.sqrt(2 * np.log(parent_visits) / visits)
            child_scores.append(exploitation + (exploration if explore else 0))
        return children[np.argmax(child_scores)]

    # Returns the unique key of a nodes state
    def get_state_key(self, state_manager, state):
        key = state_manager.get_state_key(state)
        if not key in self.wins_visits.keys():
            self.wins_visits[key] = [0, 0]
        return key

    # Returns the number of wins for a state
    def get_wins(self, state_manager, state):
        return self.wins_visits[self.get_state_key(state_manager, state)][0]

    # Returns the number of visits for a state
    def get_visits(self, state_manager, state):
        return self.wins_visits[self.get_state_key(state_manager, state)][1]

    # Returns the number of wins and visits for a state
    def get_wins_visits(self, state_manager, state):
        return self.wins_visits[self.get_state_key(state_manager, state)]

    # Increases the number of wins for a state
    def increase_wins(self, state_manager, state, wins):
        self.wins_visits[self.get_state_key(state_manager, state)][0] += wins

    # Increases the number of visits for a state
    def increase_visits(self, state_manager, state, visits):
        self.wins_visits[self.get_state_key(state_manager, state)][1] += visits

    # Helper function to print out tree
    def tree_printer(self, state_manager, children, tab, max_depth):
        if tab == "\t"*max_depth:
            return
        if children:
            for child in children:
                print("\n{0}State: Current Player Index: {1} - Remaining Stones: {2}"
                    .format(tab, child.state.current_player_idx, child.state.remaining_stones))
                parent_visits = self.get_visits(state_manager, child.parent.state)
                wins, visits =  self.get_wins_visits(state_manager, child.state)
                q = 0
                uct = np.Infinity
                if visits:
                    q = wins/visits
                    uct = q + np.sqrt(2 * np.log(parent_visits) / visits)
                print("{0}Wins: {1}".format(tab, wins))
                print("{0}Visits: {1}".format(tab, visits))
                print("{0}Q(s) - Value: {1}".format(tab, q))
                print("{0}UCT(s) - Value: {1}".format(tab, uct))
                self.tree_printer(state_manager, child.children, tab + str("\t"), max_depth)

    # Node class used in tree
    class Node():
        def __init__(self, state, action, parent):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []