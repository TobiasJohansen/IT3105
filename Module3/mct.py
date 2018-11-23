import numpy as np
import random

class MCT():
    def __init__(self, statemanager):
        self.statemanager = statemanager
        self.root = self.Node(statemanager.state)

    def update_root(self, new_root):
        self.root = new_root
        self.root.parent = None

    def find_leaf(self):
        node = self.root 
        children = node.children
        while children:
            # If all children have been visited, apply tree policy,
            # else randomly select and return one of the unvisited
            if all(child.visits > 0 for child in children):
                node = self.max_uct(children, node.visits)
                children = node.children
            else:
                unvisited_children = [child for child in children if child.visits == 0]
                return random.choice(unvisited_children)
        # Generate all child nodes of the current node.
        child_states = self.statemanager.gen_children(node.state)
        for child_state in child_states: 
            node.children.append(self.Node(child_state, node))
        return node if not child_states else random.choice(node.children)
    
    def max_uct(self, children, visits):
        utc_scores = [self.utc(child.wins, child.visits, visits) for child in children]
        return children[np.argmax(utc_scores)]
    
    def utc(self, wins, visits, parent_visits):
        return wins / visits + np.sqrt(2 * np.log(parent_visits) / visits)
    
    def rollout(self, leaf, anet, e=0.5):
        start_state = leaf.state
        current_state = start_state
        while not self.statemanager.final_state(current_state):
            # Makes random moves with probability e and uses ANET with probability 1-e 
            if random.random() > e:
                binarized_current_state, action_mask = self.statemanager.gen_case(current_state)
                outputs = anet([binarized_current_state], [action_mask]).detach().numpy().squeeze()
                current_state = self.statemanager.gen_child(np.argmax(outputs), current_state)
            else:
                current_state = self.statemanager.gen_random_child(current_state)
        return self.statemanager.win(start_state, current_state)

    def backpropagate(self, leaf, win):
        leaf.wins += 1 if win else 0
        leaf.visits += 1
        node = leaf.parent
        while node:
            win = not win
            node.wins += 1 if win else 0
            node.visits += 1
            node = node.parent

    class Node():
        def __init__(self, state=None, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0