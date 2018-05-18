from Environnement import Env
from Samplers import BasicSampler, ParentSampler
from NNs import FFIndiv
from collections import namedtuple
from Archive import Archive

import es
import numpy as np


Sample = namedtuple('Sample', ('params', 'score'))
Theta = namedtuple('Theta', ('mean', 'inv_cov'))

env = Env("Pendulum-v0")
act_space = env.get_action_space()
obs_space = env.get_obs_space()

n_train = 1000
n_test = 10
pop_size = 100

print("Act space:", act_space)
print("Obs space:", obs_space)

# optimization stuff
nn = FFIndiv(obs_space, act_space, hidden_size=8)
optimizer = es.OpenES(nn.get_params().shape[0], popsize=pop_size)

# archives
sample_archive = Archive(max_size=n_train * pop_size)
thetas_archive = Archive(max_size=n_train)
mu, inv_cov = optimizer.get_distrib_params()
thetas_archive.add_sample(Theta(mu, inv_cov))

# sampler
sampler = ParentSampler(sample_archive, thetas_archive, alpha=0.1)

# training
for i in range(n_train):

    batch, n_reused, scores_reused = sampler.ask(pop_size, optimizer)
    scores = np.zeros(optimizer.popsize)

    # reused samples
    for j in range(n_reused):

        scores[j] = scores_reused[j]
        # TBD add sample with different theta

    # newly drawn samples
    for j in range(n_reused, pop_size):

        nn.set_params(batch[j])
        score = env.eval(nn, render=False)
        sample_archive.add_sample(Sample(batch[j], score))
        scores[j] = score

    # statistics on the current batch
    # print(sample_archive.get_size())
    print("Best/Average score in pop {0}: {1:.2f}, {2:.2f}".format(
        i, np.max(scores), np.mean(scores)),
        "; Reused {} samples".format(n_reused))

    # optimization step
    optimizer.tell(batch, scores)
    mu, inv_cov = optimizer.get_distrib_params()
    thetas_archive.add_sample(Theta(mu, inv_cov))

# testing best
best_params = optimizer.best_param()
nn.set_params(best_params)
for i in range(n_test):
    score = env.eval(nn, render=True)
    print(score)

env.close()
