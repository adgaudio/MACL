from engine.api.cartesian import *
import numpy as np

#  Example 1: Visualizing an agent in a 2d grid
env = CartesianEnvironment((200, 200))

a1 = A.RandomAgent(n_coord=2, walk=True)
env.add_agent(a1.ID, n_coord=2)
a2 = A.DeterministicAgent(np.random.randint(-1, 2, (1000, 1, 2)))
env.add_agent(a2.ID)


def run_agents(a1, a2, env):
    a1.next_action(env)
    a2.next_action(env)

with cartesian_display(env, scale=(3, 3)) as d:
    d.schedule(run_agents, None, a1, a2, env)


#  Example 2: Two concurrent (and independent) visualizations
#  This creates two windows.  Note that we pick up the last environment from
#  where we left off.

env20 = CartesianEnvironment((100, 100))
a10 = A.SeekAnyLeastVisitedAgent()
env20.add_agent(a10.ID, n_coord=25)

with many_displays(
        cartesian_display(env20, refresh_interval=1 / 20, scale=(5, 5)),
        cartesian_display(env, refresh_interval=1 / 30, scale=(3, 3))) \
        as (d, d20):
    d.schedule(run_agents, None, a1, a2, env)
    d20.schedule(a10.next_action, None, env20)
