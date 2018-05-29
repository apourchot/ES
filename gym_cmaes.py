from Environnement import Env
from NNs import FFIndiv
from collections import namedtuple
from Archive import Archive

import ES
import numpy as np
import Samplers


Sample = namedtuple('Sample', ('params', 'score', 'gens'))
Theta = namedtuple('Theta', ('mean', 'cov'))

env = Env("CartPole-v0")
act_space = env.get_action_space()
obs_space = env.get_obs_space()

n_train = 1000
n_test = 10
pop_size = 100
curr_gen = 0

print("Act space:", act_space)
print("Obs space:", obs_space)

# optimization stuff
nn = FFIndiv(obs_space, act_space, hidden_size=16)
optimizer = ES.CMAES(nn.get_params().shape[0], pop_size=pop_size)

# archives
sample_archive = Archive(max_size=n_train * pop_size)
thetas_archive = Archive(max_size=n_train)

# sampler
sampler = Samplers.BasicSampler(
    sample_archive, thetas_archive)

# training
for i in range(n_train):

    batch = optimizer.ask(pop_size)
    scores = np.zeros(pop_size)
    # newly drawn samples
    for j in range(pop_size):

        nn.set_params(batch[j])
        score = env.eval(nn, render=False)
        sample_archive.add_sample(Sample(batch[j], score, [curr_gen]))
        scores[j] = score

    # statistics on the current batch
    print("Best/Average score in pop {0}: {1:.2f}, {2:.2f}".format(
        i, np.max(scores), np.mean(scores)))

    # optimization step
    optimizer.tell(batch, scores)
    curr_gen += 1

# testing best
best_params = optimizer.best_param()
nn.set_params(best_params)
for i in range(n_test):
    score = env.eval(nn, render=True)
    print(score)

env.close()
