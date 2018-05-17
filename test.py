from Environnement import Env
from NNs import FFIndiv
from Samplers import BasicSampler

from time import time

import numpy as np
import scipy.stats as sps

env = Env("Acrobot-v1")
act_space = env.get_action_space()
obs_space = env.get_obs_space()

print("Act space:", act_space)
print("Obs space:", obs_space)

nn = FFIndiv(obs_space, act_space)
sampler = BasicSampler(0, 0, nn.get_params())

a = time()
pop = sampler.sample()
grad = sampler.grad_p

print(grad[0][:10])
print(pop[0][:10])
print(nn.get_params()[:10])
print(time()-a)

for ind in pop:
    nn.set_params(ind)
    print(env.eval(nn, render=False, t_max=1000))
