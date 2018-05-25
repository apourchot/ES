import numpy as np
import operator

from scipy.stats import multivariate_normal
from scipy.stats import norm


class BasicSampler():

    """
    Simple sampler that relies on the ask method
    of the optimizers
    """

    def __init__(self, sample_archive, thetas_archive,
                 accept_ratio=0):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        return

    def ask(self, pop_size, optimizer):
        return optimizer.ask(pop_size), 0, 0, 0


class ParentSamplerES():

    """
    Importance mixing with previous population optimized for
    isotropic covariance matrices
    """

    def __init__(self, sample_archive, thetas_archive,
                 accept_ratio=0.5):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.alpha = 1 - accept_ratio

    def ask(self, pop_size, optimizer):

        if len(self.sample_archive) < pop_size:
            return optimizer.ask(pop_size), 0, [], []

        # old stuff
        old_thetas = self.thetas_archive[-2]
        old_mean = old_thetas.mean
        old_cov = old_thetas.cov

        def old_pdf(z):
            return norm.pdf(z, loc=old_mean, scale=old_cov).sum()

        # new stuff
        thetas = self.thetas_archive[-1]
        mean = thetas.mean
        cov = thetas.cov

        def new_pdf(z):
            return norm.pdf(z, loc=mean, scale=cov).sum()

        old_batch = self.sample_archive[-pop_size:]
        batch = np.zeros((len(old_batch), mean.shape[0]))

        # rejection sampling
        cpt = 0
        scores_reused = []
        idx_reused = []
        for i in range(pop_size):

            sample = old_batch[i]
            params = sample.params
            u = np.random.uniform(0, 1)

            if u < (1 - self.alpha) * new_pdf(params) / old_pdf(params):
                batch[cpt] = params
                scores_reused.append(sample.score)
                idx_reused.append(len(self.sample_archive) - pop_size + i)
                cpt += 1

        n_reused = cpt

        # inverse rejection sampling
        while cpt < pop_size:

            params = optimizer.ask(1).reshape(-1)
            u = np.random.uniform(0, 1)

            if u < self.alpha:
                batch[cpt] = params
                cpt += 1

            elif u < 1 - old_pdf(params) / new_pdf(params):
                batch[cpt] = params
                cpt += 1

        return batch, n_reused, idx_reused, scores_reused


class ParentSampler():

    """
    Importance mixing with previous population
    """

    def __init__(self, sample_archive, thetas_archive,
                 accept_ratio=0.5):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.alpha = 1 - accept_ratio

    def ask(self, pop_size, optimizer):

        if len(self.sample_archive) < pop_size:
            return optimizer.ask(pop_size), 0, [], []

        # old stuff
        old_thetas = self.thetas_archive[-2]
        old_mean = old_thetas.mean
        old_cov = old_thetas.cov

        def old_pdf(z):
            return multivariate_normal.pdf(z, old_mean,
                                           old_cov)

        # new stuff
        thetas = self.thetas_archive[-1]
        mean = thetas.mean
        cov = thetas.cov

        def new_pdf(z):
            return multivariate_normal.pdf(z, mean, cov)

        # not really correct but meh
        old_batch = self.sample_archive[-pop_size:]
        batch = np.zeros((len(old_batch), mean.shape[0]))

        # rejection sampling
        cpt = 0
        scores_reused = []
        idx_reused = []
        for i in range(pop_size):

            sample = old_batch[i]
            params = sample.params
            u = np.random.uniform(0, 1)

            if u < (1 - self.alpha) * new_pdf(params) / old_pdf(params):
                batch[cpt] = params
                scores_reused.append(sample.score)
                idx_reused.append(len(self.sample_archive) - pop_size + i)
                cpt += 1

        n_reused = cpt

        # inverse rejection sampling
        while cpt < pop_size:

            params = optimizer.ask(1).reshape(-1)
            u = np.random.uniform(0, 1)

            if u < self.alpha:
                batch[cpt] = params
                cpt += 1

            elif u < 1 - old_pdf(params) / new_pdf(params):
                batch[cpt] = params
                cpt += 1

        return batch, n_reused, idx_reused, scores_reused


class BestAncestorSamplerES():

    """
    Using importance mixing on the previous "most likely"
    samples from the archive, optimized for isotropic 
    covariance matrices
    """

    def __init__(self, sample_archive, thetas_archive,
                 accept_ratio=0.5):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.accept_ratio = accept_ratio
        self.alpha = 1

    def ask(self, pop_size, optimizer):

        if len(self.sample_archive) < pop_size:
            return optimizer.ask(pop_size), 0, [], []

        # new stuff
        thetas = self.thetas_archive[-1]
        mean = thetas.mean
        cov = thetas.cov

        def new_pdf(z):
            return norm.pdf(z, loc=mean, scale=cov).sum()

        ratios = {}
        for i in range(len(self.sample_archive)):

            # sample
            sample = self.sample_archive[i]
            params = sample.params

            # all precedent distributions
            for j in range(len(sample.gens)):

                # parameters
                old_thetas = self.thetas_archive[sample.gens[j]]
                old_mean = old_thetas.mean
                old_cov = old_thetas.cov

                r = new_pdf(params) / norm.pdf(params,
                                               loc=old_mean,
                                               scale=old_cov).sum()
                ratios[(i, j)] = r

        sorted_ratios = sorted(ratios.items(),
                               key=operator.itemgetter(1))[-pop_size:]
        print(sorted_ratios)
        batch = np.zeros((pop_size, mean.shape[0]))

        cpt = 0
        scores_reused = []
        idx_reused = []
        to_draw = []

        self.alpha = 1 - self.accept_ratio / np.mean(sorted_ratios[:, 1])

        # rejection sampling
        for i in range(pop_size):

            (id_sample, gen), ratio = sorted_ratios[i]
            sample = self.sample_archive[id_sample]
            params = sample.params
            u = np.random.uniform(0, 1)

            if u < (1 - self.alpha) * ratio:
                batch[cpt] = params
                scores_reused.append(sample.score)
                idx_reused.append(id_sample)
                cpt += 1

            else:
                to_draw.append(sample.gens[gen])

        n_reused = cpt

        # inverse rejection sampling
        while len(to_draw) > 0:

            params = optimizer.ask(1).reshape(-1)
            gen = to_draw[-1]
            sample = self.sample_archive[id_sample]
            old_thetas = self.thetas_archive[gen]
            old_mean = old_thetas.mean
            old_cov = old_thetas.cov

            def old_pdf(z):
                return norm.pdf(z, loc=old_mean, scale=old_cov).sum()

            u = np.random.uniform(0, 1)
            if u < self.alpha:
                batch[cpt] = params
                to_draw.pop()
                cpt += 1

            elif u < 1 - old_pdf(params) / new_pdf(params):
                batch[cpt] = params
                to_draw.pop()
                cpt += 1

        return batch, n_reused, idx_reused, scores_reused


class BestAncestorSampler():

    """
    Using importance mixing on the previous "most likely"
    samples from the archive
    """

    def __init__(self, sample_archive, thetas_archive,
                 accept_ratio=0.5):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.accept_ratio = accept_ratio
        self.alpha = 1

    def ask(self, pop_size, optimizer):

        if len(self.sample_archive) < pop_size:
            return optimizer.ask(pop_size), 0, [], []

        # new stuff
        thetas = self.thetas_archive[-1]
        mean = thetas.mean
        cov = thetas.cov

        def new_pdf(z):
            return multivariate_normal.pdf(z, mean, cov)

        ratios = {}
        for i in range(len(self.sample_archive)):

            # sample
            sample = self.sample_archive[i]
            params = sample.params

            # all precedent distributions
            for j in range(len(sample.gens)):

                # parameters
                old_thetas = self.thetas_archive[sample.gens[j]]
                old_mean = old_thetas.mean
                old_cov = old_thetas.cov

                r = new_pdf(params) / \
                    multivariate_normal.pdf(params, old_mean, old_cov)
                ratios[(i, j)] = r

        sorted_ratios = sorted(ratios.items(),
                               key=operator.itemgetter(1))[-pop_size:]
        batch = np.zeros((pop_size, mean.shape[0]))

        cpt = 0
        scores_reused = []
        idx_reused = []
        to_draw = []

        # rejection sampling
        for i in range(pop_size):

            (id_sample, gen), ratio = sorted_ratios[i]
            sample = self.sample_archive[id_sample]
            params = sample.params
            u = np.random.uniform(0, 1)

            self.alpha = 1 - self.accept_ratio / ratio

            if u < (1 - self.alpha) * ratio:
                batch[cpt] = params
                scores_reused.append(sample.score)
                idx_reused.append(id_sample)
                cpt += 1

            else:
                to_draw.append(sample.gens[gen])

        n_reused = cpt

        # inverse rejection sampling
        while len(to_draw) > 0:

            params = optimizer.ask(1).reshape(-1)
            gen = to_draw[-1]
            old_thetas = self.thetas_archive[gen]
            old_mean = old_thetas.mean
            old_cov = old_thetas.cov

            def old_pdf(z):
                return multivariate_normal.pdf(z, old_mean, old_cov)

            u = np.random.uniform(0, 1)
            if u < self.alpha:
                batch[cpt] = params
                to_draw.pop()
                cpt += 1

            elif u < 1 - old_pdf(params) / new_pdf(params):
                batch[cpt] = params
                to_draw.pop()
                cpt += 1

        return batch, n_reused, idx_reused, scores_reused


class ClosestSampler():

    """
    Using the closest samples found in 
    the archive
    """

    def __init__(self, sample_archive, thetas_archive,
                 accept_ratio=0.75):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.accept_ratio = accept_ratio

    def ask(self, pop_size, optimizer):

        if len(self.sample_archive) < pop_size:
            return optimizer.ask(pop_size), 0, [], []

        thetas = self.thetas_archive[-1]
        mean = thetas.mean
        cov = thetas.cov
        if np.isscalar(cov):
            cov = cov * np.eye(mean.shape[0])
        n_reused = int(self.accept_ratio * pop_size)

        dists = []
        for i in range(len(self.sample_archive)):

            sample = self.sample_archive[i]
            params = sample.params
            z = params - mean
            dists.append(z.T @ np.linalg.inv(cov) @ z)

        idx_sorted = np.argsort(dists)
        batch = np.zeros((pop_size, mean.shape[0]))
        idx_reused = []
        scores_reused = []

        for i in range(n_reused):

            idx = idx_sorted[i]
            sample = self.sample_archive[idx]
            params = sample.params
            batch[i] = sample.params

            # biased
            scores_reused.append(sample.score)
            idx_reused.append(idx)

        for i in range(n_reused, pop_size):

            params = optimizer.ask(1).reshape(-1)
            batch[i] = params

        return batch, n_reused, idx_reused, scores_reused
