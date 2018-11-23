import numpy as np
import random

class Statemanager():

    neighbors = [[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0]]

    def init_state(self, n, p=1):
        self.state = self.State([0]*n**2, p, False)

    def final_state(self, state=None):
        if not state:
            state = self.state
        return True if state.winner else False

    # Generate a random legal child
    def gen_random_child(self, state=None):
        if not state:
            state = self.state
        child = None
        while not child:
            child = self.gen_child(random.choice(range(len(state.board))), state)
        return child
    
    # Generates the child which is a result of performing action a on state
    def gen_child(self, i, state=None):
        if not state:
            state = self.state
        board = state.board
        player = state.player
        new_player = [1,2][player%2]
        if board[i] == 0:
            new_board = board[:]
            new_board[i] = player
            return self.State(new_board, new_player, self.winner(new_board))
        else: 
            return None
    
    # Generates all legal children
    def gen_children(self, state=None):
        if not state:
            state = self.state
        children = []
        if self.final_state(state):
            return children
        for i in range(len(state.board)):
            child = self.gen_child(i, state)
            if child:
                children.append(child)
        return children

    # Performs a search from the upper part of the board to the lower part of the board to
    # find a winner returns None if game is not over yet
    def winner(self, board=None):
        if not board:
            board = self.state.board
        n = int(np.sqrt(len(board)))
        # Check win for both players
        for player in [1,2]:
            # Lists containing spaces to visit and spaces that have been visited
            explore = []
            explored = []
            # The indexer helps select the first spaces to be examined based on the player
            # If the player is 1, it selects the spaces on the upper left side /
            # If the player is 2, it selects the spaces on the upper right side \
            indexer = (lambda i, n : i*n) if player == 1 else (lambda i, n : i)
            for i in range(n):
                index = indexer(i,n)
                if board[index] == player:
                    explore.append([index // n, index % n]) 
            # While there still are spaces to explore:
            while explore:
                # Remove the space from explore and add it to explored
                index = explore.pop()
                explored.append(index)
                # Find all neighbors
                neighbor_coordinates = [np.array(index) + neighbor for neighbor in self.neighbors]
                # Check all if any neighbors have been selected by player, hasn't been explored
                # and exists. If so, check if they are connected to the lower part of the board, 
                # winning the player the game. If not, its simply added to the list of spaces to explore.
                for neighbor_coordinate in neighbor_coordinates:
                    x, y = neighbor_coordinate
                    if 0 <= x < n and 0 <= y < n and board[x*n+y] == player and [x, y] not in explored:
                        if [y,x][player-1] == n-1:
                                return player
                        explore.append([x,y])
        return None

    # Returns a binarized state and a masking for the legal moves 
    def gen_case(self, state=None):
        if not state:
            state = self.state
        binarized_state = []
        for board_space in state.board:
            binarized_state.extend([[0,0],[1,0],[0,1]][board_space])
        binarized_state.extend([[1,0],[0,1]][state.player-1])
        return binarized_state, [1 if i == 0 else 0 for i in state.board]    

    def win(self, start_state, final_state):
        return start_state.player != final_state.winner    

    def print_state(self, state=None):
        if not state:
            state = self.state
        board = state.board
        n = int(np.sqrt(len(board)))
        diamond = [[" "]*(2*n-1) for _ in range(2*n-1)]
        for i in range(n):
            for j in range(n):
                diamond[i+j][n-1+j-i] = board[i*n+j]
        for row in diamond:
            string = ""
            for element in row:
                string += str(element)
            print(string)
        print()

    class State():
        def __init__(self, board, player, winner):
            self.board = board
            self.player = player
            self.winner = winner

        def __eq__(self, other): 
            return self.__dict__ == other.__dict__