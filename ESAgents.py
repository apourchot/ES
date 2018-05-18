import numpy as np

from collections import namedtuple
Sample = namedtuple('Sample', ('params', 'scores', 'thetas', 'goals'))


class VanillaES():

    def __init__(self, env, sample_archive, thetas_archive,
                 nn_class, observation_space, action_space,
                 sampler_class, lr=10**-3, batch_size=100):

        # misc
        self.batch_size = batch_size
        self.best_score = 0
        self.n_batch = 0
        self.lr = lr

        # env and archives
        self.env = env
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive

        # nn
        self.nn = nn_class(observation_space, action_space)
        self.best_params = self.nn.get_params()

        # sampling strategy
        self.sampler = sampler_class(sample_archive, thetas_archive)
        self.mean = self.nn.get_params()
        self.cov = np.eye(self.mean.shape[0])

        # thetas_archive.add_sample(sampler.get_params())

    def train(self):
        """
        Train the parameters of the distribution
        by generating a new population and computing
        the corresponding gradient
        """

        # get the new generation according to the
        # given sampling strategy
        batch = self.sampler.sample(self.batch_size, self.mean, self.cov)

        # evaluate the fitness of the individuals
        # and add the samples to the archive
        scores = []
        for ind in batch:
            self.nn.set_params(ind)
            score = self.env.eval(self.nn, render=False)

            if score > self.best_score:
                self.best_score = score
                self.best_params = ind

            scores.append(score)
            # samples.append(Sample(ind, score, 0, .n_batch]))

        # apply fitness shaping to the scores
        fitness = np.zeros(self.batch_size)
        rank = np.argsort(scores)
        for i in range(self.batch_size):
            fitness[rank[i]] = 2 * i / (self.batch_size - 1) - 1

        # update the sampler based on last sampled individuals
        grad = self.compute_grad(batch, fitness)
        self.mean += self.lr * grad
        self.n_batch += 1

        print(self.mean[:10])

        # append to the archives
        # sample_archive.add_sampales(samples)
        # thetas_archive.add_samples(sampler.get_params())

        print("Best/Average score in pop {0}: {1:.2f}, {2:.2f}".format(
            self.n_batch, np.max(scores), np.mean(scores)))

    def compute_grad(self, batch, scores):
        """
        Computes the gradient
        """

        grad_p = batch - self.mean
        scores = np.array(scores).reshape(-1, 1)
        grad = 1 / self.batch_size * grad_p.T @ scores
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
