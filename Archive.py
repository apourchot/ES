import numpy as np


class Archive():

    def __init__(self, max_size=1000):

        # max size so no overflow
        self.max_size = max_size
        self.cpt = 0

        # content of the archive
        self.samples = []

    def add_sample(self, sample):

        # adding the sample to the archive
        if cpt < max_size:
            self.samples.append(sample)
            cpt += 1

        else:
            self.samples[cpt%max_size] = sample

    def add_samples(self, samples):

        # adding the samples to the archive
        for sample in samples:
            self.add_sample(sample)
