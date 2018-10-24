from games.nim.game import Game
from games.nim.statemanager import StateManager
from games.player import Player
from src.mcts import MCTS

play = True
while play:

    # Create the brain of the MCTS AI(s), the list of Players, set up the Game and initialize the StateManager
    brain = MCTS(m=100, rollout_batch_size=1)
    players = [Player("Tobias"), Player("AI-bert", brain=brain)]
    game = Game(players, starting_player_idx=0, total_nr_of_stones=10, max_selection=3)
    state_manager = StateManager(game)
    
    # Play until game is over
    while not game.over:
        # Update the internal state of the AI(s) brain
        brain.update(game.state)
        # Request action from current player and execute it
        game.select_stones(game.current_player().take_turn(game, state_manager))
    
    # Ask user if a new game should be started
    input_given = False
    while not input_given:
        inpt = input("Do you want to play another game of NIM? (y/n): ")
        if inpt == "y":
            input_given = True
        elif inpt == "n": 
            input_given = True
            play = False