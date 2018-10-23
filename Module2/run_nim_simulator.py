from games.nim.simulator import Simulator

batch_parameters = {
    "g": 2,
    "p": 0,
    "m": 80,
    "n": 10,
    "k": 3,
    "verbose": True
}

sim = Simulator(player_names=["AI-bert", "AI-na"])
sim.simulate_batch(**batch_parameters)