import numpy as np


class BasicSampler():

    def __init__(self, sample_archive, thetas_archive):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        return

    def ask(self, batch_size, optimizer):
        return optimizer.ask()


class ParentSampler():

    def __init__(self, sample_archive, thetas_archive,
                 alpha=0.01):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.alpha = alpha

    def ask(self, batch_size, optimizer):

        if len(self.sample_archive) < batch_size:
            return optimizer.ask(batch_size), 0, []

        # old stuff
        old_thetas = self.thetas_archive[-2]
        old_mean = old_thetas.mean
        old_inv_cov = old_thetas.inv_cov
        old_inv_det = np.linalg.det(old_inv_cov)

        def old_pdf(z):
            x = z - old_mean
            return np.exp(-0.5 * x.T@old_inv_cov@x) * np.sqrt(old_inv_det)

        # new stuff
        thetas = self.thetas_archive[-2]
        mean = thetas.mean
        inv_cov = thetas.inv_cov
        inv_det = np.linalg.det(inv_cov)

        def new_pdf(z):
            x = z - mean
            return np.exp(-0.5 * x.T@inv_cov@x) * np.sqrt(inv_det)

        old_batch = self.sample_archive[-batch_size:]
        batch = np.zeros((len(old_batch), mean.shape[0]))

        # rejection sampling
        cpt = 0
        scores_reused = []
        for i in range(batch_size):

            sample = old_batch[i]
            params = sample.params
            u = np.random.uniform(0, 1)

            if u < (1 - self.alpha) * new_pdf(params) / old_pdf(params):
                batch[cpt] = params
                scores_reused.append(sample.score)
                cpt += 1

        n_reused = cpt

        # inverse rejection sampling
        while cpt < batch_size:

            params = optimizer.ask(1).reshape(-1)
            u = np.random.uniform(0, 1)

            if u < self.alpha:
                batch[cpt] = params
                cpt += 1

            elif u < 1 - old_pdf(params) / new_pdf(params):
                batch[cpt] = params
                cpt += 1

        return batch, n_reused, scores_reused
