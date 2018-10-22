class Player():
    def __init__(self, name, brain=None):
        self.name = name
        self.brain = brain

    def take_turn(self, game, state_manager=None):
        return self.brain.get_action(state_manager) if self.brain else game.user_input()