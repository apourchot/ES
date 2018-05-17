import numpy as np
from scipy.linalg import cholesky


class GaussianSampler():

    def __init__(self, sample_archive, thetas_archive):
        """
        Simple gaussian sampler with no sample reuse
        """

        # misc
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive

    def sample(self, batch_size, mean, cov):

        pop = np.zeros((batch_size, self.mean.shape[0]))
        L = cholesky(cov)
        for i in range(self.mean.shape[0]):
            tmp = L@np.random.rand(self.pop_size//2)
            pop[:self.pop_size//2,i] = self.mean[i] + tmp
            pop[self.pop_size//2:,i] = self.mean[i] - tmp
            self.grad_p[:self.pop_size//2,i] =  tmp
            self.grad_p[self.pop_size//2:,i] = -tmp

        self.grad_p = np.zeros(pop.shape)
        self.grad_p = pop - self.mean
        return pop

    # def update_params(self, scores):
    # 
    #     scores = np.array(scores).reshape(-1, 1)
    #     grad = 1/(self.pop_size*self.sigma) * self.grad_p.T @ scores
    #     grad = grad.flatten()
    #     self.mean += self.lr * grad
