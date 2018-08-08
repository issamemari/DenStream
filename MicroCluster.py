import numpy as np

class MicroCluster():

    def __init__(self, lamb):

        self.lamb = lamb
        self.weight = 0
        self.instance_count = 0
        self.reduction_factor = 2 ** (-lamb)

    def insert_sample(self, sample):

        if self.instance_count == 0:
            self.dimensions = sample.size
            self.weighted_linear_sum = np.zeros(self.dimensions)
            self.weighted_squared_sum = np.zeros(self.dimensions)
            self.center = np.zeros(self.dimensions)
            self.radius = 0

        # Update weight
        self.weight *= self.reduction_factor
        self.weight += 1

        # Update the weighted linear sum of instances
        self.weighted_linear_sum *= self.reduction_factor
        self.weighted_linear_sum += sample

        # Update the weighted squared sum of instances
        self.weighted_squared_sum *= self.reduction_factor
        self.weighted_squared_sum += sample ** 2

        # Update micro-cluster center
        self.center = self.weighted_linear_sum / self.weight

        # Update micro-cluster radius
