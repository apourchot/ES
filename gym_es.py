from Environnement import Env
from NNs import FFIndiv
from collections import namedtuple
from Archive import Archive
from time import time

import ES
import numpy as np
import pandas as pd
import Samplers
import datetime
import pathlib

n_experiments = 5
n_train = 100
n_test = 1
pop_size = 100
accept_ratio = 0.7
now = datetime.datetime.now()

for n in range(n_experiments):

    Sample = namedtuple('Sample', ('params', 'score', 'gens'))
    Theta = namedtuple('Theta', ('mean', 'cov', 'samples'))

    curr_gen = 0
    n_ind = 0

    env = Env("CartPole-v0")
    act_space = env.get_action_space()
    obs_space = env.get_obs_space()

    # optimization stuff
    nn = FFIndiv(obs_space, act_space, hidden_size=2)
    best_params = nn.get_params()
    best_score = env.eval(nn, render=False)
    optimizer = ES.CMAES(
        nn.get_params().shape[0], pop_size=pop_size)

    # archives
    sample_archive = Archive(max_size=n_train * pop_size)
    thetas_archive = Archive(max_size=n_train)
    mu, cov = optimizer.get_distrib_params()
    thetas_archive.add_sample(Theta(mu, cov, []))

    # sampler
    sampler = Samplers.ClosestSampler(
        sample_archive, thetas_archive, accept_ratio=accept_ratio)

    print("Problem dimension:", mu.shape[0])
    df = pd.DataFrame(columns=["n_reused",
                               "best_score",
                               "average_score",
                               "sample_time",
                               "evaluation_time"])

    # training
    for i in range(n_train):

        # print(thetas_archive)
        t_sample = time()
        batch, n_reused, idx_reused, scores_reused = sampler.ask(
            pop_size, optimizer)
        scores = np.zeros(pop_size)
        t_sample = time() - t_sample

        # reused samples
        for j in range(n_reused):

            scores[j] = scores_reused[j]
            sample_archive.add_gen(idx_reused[j], curr_gen)
            thetas_archive[curr_gen].samples.append(idx_reused[j])

        # newly drawn samples
        t_eval = time()
        for j in range(n_reused, pop_size):

            nn.set_params(batch[j])
            score = env.eval(nn, render=False)
            sample_archive.add_sample(Sample(batch[j], score, [curr_gen]))
            thetas_archive[curr_gen].samples.append(n_ind)
            scores[j] = score
            n_ind += 1  

            if score > best_score:
                best_params = batch[j]
        t_eval = time() - t_eval

        # statistics on the current batch
        print("Best/Average score in pop {0}: {1:.2f}, {2:.2f}".format(
            i, np.max(scores), np.mean(scores)),
            "; Reused {} samples".format(n_reused))

        # optimization step
        optimizer.tell(batch, scores)
        mu, cov = optimizer.get_distrib_params()
        thetas_archive.add_sample(Theta(
            mu, cov, []))
        curr_gen += 1

        # adding to dataframe
        df = df.append({"n_reused": n_reused,
                        "best_score": np.max(scores),
                        "average_score": np.mean(scores),
                        "sample_time": t_sample,
                        "evaluation_time": t_eval},
                       ignore_index=True)

    # testing best
    nn.set_params(best_params)
    for i in range(n_test):
        score = env.eval(nn, render=True)
        print(score)

    dirname = "gym_experiments/" + env.get_name() + "_" + optimizer.__class__.__name__ + "_" + \
        sampler.__class__.__name__ + "_alpha_" + str(accept_ratio) + "_popsize_" + str(pop_size) + "_" +\
        str(now.hour) + "h" + str(now.minute) + "m" + str(now.second) + 's'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    filename = dirname + "/exp_" + str(n + 1) + "_" + str(n_experiments)\
        + ".pick"
    df.to_pickle(filename)

    env.close()
