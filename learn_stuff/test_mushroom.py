from mushroom_rl.environments import GridWorld

from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

from mushroom_rl.algorithms.value import QLearning

from mushroom_rl.core.core import Core


'''
    Notes for using MushroomRL:
    ============================
        - GridWorld is the environment (contains the Markov Decision Process)

        - policy is the algorithm that the agent follows 
            - this is what contains the Parameters to Optimize
            - formally it is a probability distribution representing the action space
            - when policy is conditioned on the observation, you get the max posterior to find the best action associated with the observation

        - agent is the one that is performing the actions and learning
            - or just consider an agent as a Value Function
            - a value function evaluates the policy and gives it the reward

        - so in reality we have a Policy (network/model) which is evaluated using the agent (value function) on an environment (Markov Decision Process)
'''


mdp = GridWorld(width=3, height=3, goal=(2,2), start=(0,0))

epsilon = Parameter(value=1.)
policy = EpsGreedy(epsilon=epsilon)

learning_rate = Parameter(value=.6)
agent = QLearning(mdp.info, policy, learning_rate)

core = Core(agent, mdp)
core.learn(n_steps=1000, n_steps_per_fit=1)

import numpy as np
shape = agent.Q.shape
q = np.zeros(shape)


for i in range(shape[0]):
    for j in range(shape[1]):
        state = np.array([i])
        action = np.array([j])
        q[i,j] = agent.Q.predict(state,action)

print(q)