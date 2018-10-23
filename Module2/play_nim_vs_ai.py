from games.nim.game import Game
from games.nim.statemanager import StateManager
from games.player import Player
from src.mcts import MCTS

play = True
while play:
    brain = MCTS(m=1000, rollout_batch_size=100)
    players = [Player("Tobias"), Player("AI-bert", brain=brain)]
    game = Game(players, 0, 15, 3)
    state_manager = StateManager(game)
    while not game.over:
        brain.update(game.state)
        game.select_stones(game.current_player().take_turn(game, state_manager))
    
    input_given = False
    while not input_given:
        inpt = input("Do you want to play another game of NIM? (y/n): ")
        if inpt == "y":
            input_given = True
        elif inpt == "n": 
            input_given = True
            play = False