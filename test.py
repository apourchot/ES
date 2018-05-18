from Environnement import Env
from NNs import FFIndiv
from Samplers import GaussianSampler

from time import time

import numpy as np

env = Env("CartPole-v0")
act_space = env.get_action_space()
obs_space = env.get_obs_space()

print("Act space:", act_space)
print("Obs space:", obs_space)

nn = FFIndiv(obs_space, act_space)
sampler = GaussianSampler(0, 0)

a = time()
batch = sampler.sample(32, nn.get_params(), np.eye(nn.get_params().shape[0]))

print(nn.get_params()[:10])
print(time() - a)

for ind in batch:
    nn.set_params(ind)
    print(env.eval(nn, render=False, t_max=1000))
