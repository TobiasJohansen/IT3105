from games.nim.simulator import Simulator

sim = Simulator(2)
score = sim.simulate_batch(10, 1, 1000, 15, 3, batch_size=10, verbose=True)
print(score)