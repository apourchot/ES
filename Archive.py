class Archive(list):

    def __init__(self, max_size=1000):

        # max size so no overflow
        self.max_size = max_size
        self.cpt = 0

    def add_sample(self, sample):

        # adding the sample to the archive
        if self.cpt < self.max_size:
            self.append(sample)
            self.cpt += 1

        else:
            self[self.cpt % self.max_size] = sample

    def add_samples(self, samples):

        # adding the samples to the archive
        for sample in samples:
            self.add_sample(sample)

    def get_size(self):
        return min(self.max_size, self.cpt)
