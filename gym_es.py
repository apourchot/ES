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
best_params = nn.get_params()
best_score = env.eval(nn, render=False)
optimizer = ES.CMAES(
    nn.get_params().shape[0], pop_size=pop_size, antithetic=False)

# archives
sample_archive = Archive(max_size=n_train * pop_size)
thetas_archive = Archive(max_size=n_train)
mu, cov = optimizer.get_distrib_params()
thetas_archive.add_sample(Theta(mu, cov))

# sampler
sampler = Samplers.ClosestSampler(
    sample_archive, thetas_archive, accept_ratio=0.8)

print("Problem dimension:", mu.shape[0])

# training
for i in range(n_train):

    batch, n_reused, idx_reused, scores_reused = sampler.ask(
        pop_size, optimizer)
    scores = np.zeros(pop_size)

    # reused samples
    for j in range(n_reused):

        scores[j] = scores_reused[j]
        sample_archive.add_gen(idx_reused[j], curr_gen)

    # newly drawn samples
    for j in range(n_reused, pop_size):

        # print(batch[j])
        nn.set_params(batch[j])
        score = env.eval(nn, render=False)
        sample_archive.add_sample(Sample(batch[j], score, [curr_gen]))
        scores[j] = score

        if score > best_score:
            best_params = batch[j]

    # statistics on the current batch
    print("Best/Average score in pop {0}: {1:.2f}, {2:.2f}".format(
        i, np.max(scores), np.mean(scores)),
        "; Reused {} samples".format(n_reused))

    # optimization step
    optimizer.tell(batch, scores)
    mu, cov = optimizer.get_distrib_params()
    thetas_archive.add_sample(Theta(mu, cov))
    curr_gen += 1

# testing best
nn.set_params(best_params)
for i in range(n_test):
    score = env.eval(nn, render=True)
    print(score)

env.close()
