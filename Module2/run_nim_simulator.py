from games.nim.simulator import Simulator

batch_parameters = {
    "g": 100,
    "p": 0,
    "m": 25,
    "n": 10,
    "k": 3,
    "verbose": False
}

sim = Simulator(player_names=["AI-bert", "AI-na"])
sim.simulate_batch(**batch_parameters)
