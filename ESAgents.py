import numpy as np
from NNs import FFIndiv

from collections import namedtuple
Sample = namedtuple('Sample', ('params', 'scores', 'thetas', 'goals'))

class VanillaES():

    def __init__(self, env, sample_archive, thetas_archive,
                nn_class, observation_space, action_space,
                sampler_class, lr=10**-3, batch_size=100):

        # misc
        self.batch_size=batch_size
        self.best_score  = 0
        self.n_pop = 0
        self.lr=lr

        # env and archives
        self.env = env
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive

        # nn
        self.nn = nn_class(observation_space, action_space)
        self.best_params = self.nn.get_params()

        # sampling strategy
        self.sampler = sampler_class(sample_archive, thetas_archive)
        self.mean = nn.get_params()
        self.cov  = np.eye(self.mean.shape[0])

        # thetas_archive.add_sample(sampler.get_params())

    def train(self):
        """
        Train the parameters of the distribution
        by generating a new population and computing
        the corresponding gradient
        """

        # get the new generation according to the
        # given sampling strategy
        pop = self.sampler.sample()

        # evaluate the fitness of the individuals
        # and add the samples to the archive
        scores = []
        for ind in pop:
            self.nn.set_params(ind)
            score = self.env.eval(self.nn, render=False)

            if score > self.best_score:
                self.best_score  = score
                self.best_params = ind

            scores.append(score)
            # samples.append(Sample(ind, score, 0, [n_pop]))

        # apply fitness shaping to the scores
        fitness  = np.zeros(self.batch_size)
        rank     = np.argsort(scores)
        for i in range(pop_size):
            fitness[rank[i]] = 2*i/(pop_size-1)-1

        # update the sampler based on last sampled individuals
        grad = self.compute_grad(pop, fitness)
        self.mean += grad[0]
        self.sampler.update_params(fitness)
        self.n_pop += 1

        # append to the archives
        # sample_archive.add_samples(samples)
        # thetas_archive.add_samples(sampler.get_params())

        print("Best/Average score in pop {0}: {1:.2f}, {2:.2f}".format(self.n_pop,
                                                               np.max(scores),
                                                               np.mean(scores)))



    def compute_grad(self, batch, scores):
        """
        Computes the gradient
        """

        grad_p = 
        scores = np.array(scores).reshape(-1, 1)
        grad = 1/self.pop_size * grad_p.T @ scores
        grad = grad.flatten()
        return grad


    def test(self, n_episodes):
        """
        Test the fitness of the best individual
        """

        self.nn.set_params(self.best_params)
        for _ in range(n_episodes):
            score = self.env.eval(self.nn, render=True)
            print(score)
