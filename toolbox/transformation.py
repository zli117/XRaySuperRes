class Standardize:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __call__(self, image):
        return (image - self.mean) / self.sd
