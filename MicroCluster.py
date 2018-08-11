import numpy as np


class MicroCluster:
    def __init__(self, lambd):
        self.decay_factor = 2 ** (-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0

    def insert_sample(self, sample, weight):
        if self.sum_of_weights != 0:
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (sample - old_mean)

            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) \
                + weight * (sample - new_mean) * (sample - old_mean)

            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:
            self.mean = sample
            self.sum_of_weights = weight

    def radius(self):
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')

    def center(self):
        return self.mean