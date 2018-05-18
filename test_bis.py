from Environnement import Env
from NNs import FFIndiv
from Samplers import GaussianSampler
from ESAgents import VanillaES

env = Env("CartPole-v0")
act_space = env.get_action_space()
obs_space = env.get_obs_space()

print("Act space:", act_space)
print("Obs space:", obs_space)

es = VanillaES(env, 0, 0,
               FFIndiv, obs_space, act_space,
               GaussianSampler, lr=1, batch_size=100)

for i in range(1000):
    es.train()

es.test(10)

env.close()
