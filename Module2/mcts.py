class MCTS():
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.root = state_manager.gen_initial_state()
    
    class Node():
        def __init__(self, state):
            self.state = state

    