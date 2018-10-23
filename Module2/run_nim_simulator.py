from games.nim.simulator import Simulator

batch_parameters = {
    "g": 1,
    "p": "Mix",
    "m": 1,
    "n": 10,
    "k": 2
}

sim = Simulator(player_names=["AI-bert", "AI-na"])
sim.simulate_batch(**batch_parameters)