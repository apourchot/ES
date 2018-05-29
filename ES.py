# copied and adapted from https://github.com/hardmaru/estool/blob/master/es.py
import numpy as np
import cma

from Optimizers import Adam


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))]
    which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES_:

    """
    Wrapper for the cmaes implemention given by pycma
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1,
                 pop_size=255,
                 weight_decay=0.01):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.pop_size = pop_size
        self.weight_decay = weight_decay
        if mu_init is None:
            self.mu_init = self.num_params * [0]
        else:
            self.mu_init = mu_init
        self.es = cma.CMAEvolutionStrategy(self.mu_init,
                                           self.sigma_init,
                                           {'popsize': self.pop_size,
                                            })

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        solutions = np.array(self.es.ask(number=pop_size))
        return solutions

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        reward_table = -np.array(scores)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward_table += l2_decay
        self.es.tell(solutions, (reward_table).tolist())

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and the covariance matrix
        """
        return self.es.mean, self.es.sigma * self.es.sigma_vec.scaling * \
            np.sqrt(self.es.dC) * self.es.gp.scales

    def result(self):
        """
        Returns best params so far, best score, current score
        and sigma
        """
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))


class OpenES:

    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,             # number of model parameters
                 mu_init=None,                 # initial mean
                 sigma_init=1,                 # initial standard deviation
                 sigma_decay=0.999,            # anneal standard deviation
                 sigma_limit=0.01,             # stop annealing if less than
                 learning_rate=0.01,           # learning rate for std
                 learning_rate_decay=0.9999,   # annealing the learning rate
                 learning_rate_limit=0.001,    # stop annealing learning rate
                 pop_size=256,                 # population size
                 antithetic=False,             # whether to use anti sampling
                 weight_decay=0.01,            # weight decay coefficient
                 rank_fitness=True,            # use rank rather than fitness
                 forget_best=True):            # forget historical best

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit

        # optimizarion stuff
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.optimizer = Adam(self, learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True

    def ask(self, pop_size):
        """
        Returns a list of candidates parameterss
        """

        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        return self.mu.reshape(1, self.num_params) + epsilon * self.sigma

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)

        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        # TBD check if ok
        epsilon = (solutions - self.mu.reshape(1,
                                               self.num_params)) / self.sigma

        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1. / (self.pop_size * self.sigma) * \
            np.dot(epsilon.T, normalized_reward)

        # updating stuff
        idx = np.argsort(reward)[::-1]
        best_reward = reward[idx[0]]
        best_mu = solutions[idx[0]]

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # optimization step
        self.optimizer.stepsize = self.learning_rate
        self.optimizer.update(-change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return self.mu, self.sigma

    def result(self):
        """
        Returns best params so far, best score, current score
        and sigma
        """
        return (self.best_mu, self.best_reward,
                self.curr_best_reward, self.sigma)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))


class PEPG:
    '''Extension of PEPG with bells and whistles.'''

    def __init__(self, num_params,            # number of model parameters
                 sigma_init=0.10,             # initial standard deviation
                 sigma_alpha=0.20,            # learning rate for std
                 sigma_decay=0.999,           # anneal standard deviation
                 sigma_limit=0.01,            # stop annealing if less than
                 sigma_max_change=0.2,        # clips adaptive sigma to 20%
                 learning_rate=0.01,          # learning rate for std
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.01,    # stop annealing learning rate
                 elite_ratio=0,               # if >0 then ignore learning_rate
                 pop_size=256,                # population size
                 average_baseline=True,       # set baseline to average
                 weight_decay=0.01,           # weight decay coefficient
                 rank_fitness=True,           # use rank rather than fitness
                 forget_best=True):           # don't keep the hist best sol

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.pop_size = pop_size
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.pop_size % 2 == 0), "Population size must be even"
            self.pop_size = int(self.pop_size / 2)
        else:
            assert (self.pop_size & 1), "Population size must be odd"
            self.pop_size = int((self.pop_size - 1) / 2)

        # option to use greedy es method to select next mu,
        # rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_pop_size = int(self.pop_size * self.elite_ratio)
        self.use_elite = False
        if self.elite_pop_size > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.pop_size * 2)
        self.mu = np.zeros(self.num_params)
        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.curr_best_mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        self.epsilon = np.random.randn(self.pop_size, self.num_params)
        self.epsilon *= self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon,
            # then negative epsilon
            epsilon = np.concatenate(
                [np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, scores):
        # input must be a numpy float array
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward_table = np.array(scores)

        if self.rank_fitness:
            reward_table = compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_pop_size]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # short hand
        epsilon = self.epsilon
        sigma = self.sigma

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            rT = (reward[:self.pop_size] - reward[self.pop_size:])
            change_mu = np.dot(rT, epsilon)
            self.optimizer.stepsize = self.learning_rate
            # adam, rmsprop, momentum, etc.
            update_ratio = self.optimizer.update(-change_mu)
            # self.mu += (change_mu * self.learning_rate) # normal SGD method

        # adaptive sigma
        # normalization
        if (self.sigma_alpha > 0):
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)
            S /= sigma.reshape(1, self.num_params)
            reward_avg = (reward[:self.pop_size] +
                          reward[self.pop_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = (np.dot(rS, S)) / \
                (2 * self.pop_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(
                change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(
                change_sigma, - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if (self.sigma_decay < 1):
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (self.learning_rate_decay < 1 and
                self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):
        # return best params so far, along with historically
        # best reward, curr reward, sigma
        return (self.best_mu, self.best_reward,
                self.curr_best_reward, self.sigma)


class SimpleGA:

    """
    Simple genetic algorithm
    """

    def __init__(self, num_params,      # number of model parameters
                 sigma_init=0.1,        # initial standard deviation
                 sigma_decay=0.999,     # anneal standard deviation
                 sigma_limit=0.01,      # stop annealing if less than this
                 pop_size=256,          # population size
                 elite_ratio=0.1,       # percentage of the elites
                 forget_best=False,     # forget the historical best elites
                 weight_decay=0.01,     # weight decay coefficient
                 ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.pop_size = pop_size

        self.elite_ratio = elite_ratio
        self.elite_pop_size = int(self.pop_size * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_pop_size, self.num_params))
        self.elite_rewards = np.zeros(self.elite_pop_size)
        self.best_param = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay

    def ask(self):
        """
        Returns a list of candidates parameters
        """
        self.epsilon = np.random.randn(
            self.pop_size, self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_pop_size)
        for i in range(self.pop_size):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(
                self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, scores):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward_table = np.array(scores)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if (not self.forget_best or self.first_iteration):
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_pop_size]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def best_param(self):
        return self.best_param

    def result(self):
        # return best params so far, along with historically
        # best reward, curr reward, sigma
        return (self.best_param, self.best_reward,
                self.curr_best_reward, self.sigma)

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.


class CMAES:

    """
    CMAES implementation adapted from
    https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1,
                 step_size_init=1,
                 pop_size=255,
                 antithetic=False):

        # distribution parameters
        self.num_params = num_params
        if mu_init is not None:
            self.mu = np.array(mu_init)
        else:
            self.mu = np.zeros(num_params)
        self.step_size = step_size_init
        self.coord = np.eye(num_params)
        self.diag = sigma_init * np.ones(num_params)
        self.cov = sigma_init * np.eye(num_params)
        self.inv_sqrt_cov = 1 / np.sqrt(sigma_init) * np.eye(num_params)
        self.p_c = np.zeros(self.num_params)
        self.p_s = np.zeros(self.num_params)
        self.antithetic = antithetic

        # selection parameters
        self.pop_size = pop_size
        self.parents = pop_size // 2
        self.weights = np.array([np.log((self.parents + 0.5) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.parents_eff = 1 / (self.weights ** 2).sum()

        # adaptation  parameters
        self.c_s = (self.parents_eff + 2) / \
            (self.num_params + self.parents_eff + 5)
        self.c_c = (4 + self.parents_eff / self.num_params) / \
            (self.num_params + 4 + 2 * self.parents_eff / self.num_params)
        self.c_1 = 2 / ((self.num_params + 1.3) ** 2 + self.parents_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.parents_eff - 2 + 1 /
                                           self.parents_eff) / ((self.num_params + 2) ** 2
                                                                + self.parents_eff))
        self.damps = 1 + 2 * \
            max(0, np.sqrt((self.parents_eff - 1) /
                           (self.num_params + 1)) - 1) + self.c_s
        self.chi = np.sqrt(self.num_params) * \
            (1 - 1 / (4 * self.num_params) + 1 / (21 * self.num_params ** 2))
        self.count_eval = 0
        self.eigen_eval = 0

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.num_params, pop_size)

        return self.mu + self.step_size * (self.coord @ np.diag(np.sqrt(self.diag)) @ epsilon).T

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = -np.array(scores)
        idx_sorted = np.argsort(scores)

        # update mean
        old_mu = self.mu
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # update evolution paths
        self.p_s = (1 - self.c_s) * self.p_s + \
            np.sqrt(self.c_s * (2 - self.c_s) * self.parents_eff) * \
            self.inv_sqrt_cov @ (self.mu - old_mu) / self.step_size

        tmp_1 = np.linalg.norm(self.p_s) / np.sqrt(self.c_s * (2 - self.c_s)) \
            <= self.chi * (1.4 + 2 / (self.num_params + 1))

        self.p_c = (1 - self.c_c) * self.p_c + \
            tmp_1 * np.sqrt(self.c_c * (2 - self.c_c) * self.parents_eff) * \
            (self.mu - old_mu) / self.step_size

        # update covariance matrix
        tmp_2 = 1 / self.step_size * \
            (solutions[idx_sorted[:self.parents]] - old_mu)

        self.cov = (1 - self.c_1 - self.c_mu) * self.cov + \
            (1 - tmp_1) * self.c_1 * self.c_c * (2 - self.c_c) * self.cov + \
            self.c_1 * np.outer(self.p_c, self.p_c) + \
            self.c_mu * tmp_2.T @ np.diag(self.weights) @ tmp_2

        # update step size
        self.step_size *= np.exp((self.c_s / self.damps) *
                                 (np.linalg.norm(self.p_s) / self.chi - 1))

        # decomposition of C
        self.cov = np.triu(self.cov) + np.triu(self.cov, 1).T
        self.diag, self.coord = np.linalg.eigh(self.cov)
        self.diag = np.real(self.diag)


        self.inv_sqrt_cov = self.coord @ np.diag(
            1 / np.sqrt(self.diag)) @ self.coord.T

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and the covariance matrix
        """
        return self.mu, self.cov

    def result(self):
        """
        Returns best params so far, best score, current score
        and sigma
        """
        pass
