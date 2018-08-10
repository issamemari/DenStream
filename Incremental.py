import numpy as np


class WeightedIncrementalStandardDeviation:

    def __init__(self):

        self.decay_factor = 1
        self.mean = None
        self.variance_acc = None
        self.sum_of_weights = 0

    def update(self, sample, weight):

        #print(
            #f"Current std is {self.std()}. Current mean is {self.mean}. Updating std with new sample {sample} with weight {weight}.")
        
        if self.sum_of_weights != 0:

            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (sample - old_mean)

            # Update variance
            old_variance_acc = self.variance_acc
            new_variance_acc = old_variance_acc * ((new_sum_of_weights - weight)
                                                   / old_sum_of_weights) \
                + weight * (sample - new_mean) * (sample - old_mean)

            self.mean = new_mean
            self.variance_acc = new_variance_acc
            self.sum_of_weights = new_sum_of_weights

        else:
            self.mean = sample
            self.variance_acc = 0
            self.sum_of_weights = weight

    def std(self):
        if self.sum_of_weights > 0:
            return np.sqrt(self.variance_acc / self.sum_of_weights)
        else:
            return float('nan')

std = WeightedIncrementalStandardDeviation()
for i in range(10):
    std.update(np.array([i, i * i]), i)

print(std.std())
print(std.mean)

l = []
for i in range(10):
    for j in range(i):
        l.append([i, i * i])

std = WeightedIncrementalStandardDeviation()
for i in range(len(l)):
    std.update(np.array(l[i]), 1)

print(std.std())
print(std.mean)