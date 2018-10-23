from games.nim.simulator import Simulator

sim = Simulator(player_names=["AI-bert", "AI-na"])
score = sim.simulate_batch(g=10, p=0, m=500, n=15, k=3, rollout_batch_size=100, verbose=True)
print("\nGames won:\n{0}".format(score))
