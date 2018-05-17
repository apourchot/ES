from Environnement import Env
from NNs import FFIndiv
from Samplers import BasicSampler
from ESAgents import BasicESAgent

from time import time

import numpy as np
import scipy.stats as sps

env = Env("Pendulum-v0")
act_space = env.get_action_space()
obs_space = env.get_obs_space()

print("Act space:", act_space)
print("Obs space:", obs_space)

es = BasicESAgent(env, 0, 0,
                  FFIndiv, obs_space, act_space,
                  BasicSampler, lr=0.01, batch_size=100)

for i in range(1000):
    es.train()

es.test(10)

env.close()
