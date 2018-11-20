from games.nim.simulator import Simulator

sim = Simulator(player_names=["AI-bert", "AI-na"])
score = sim.simulate_batch(g=3, p=0, m=1000, n=99, k=6, rollout_batch_size=1, verbose=False)
print("\nGames won:\n{0}".format(score))
