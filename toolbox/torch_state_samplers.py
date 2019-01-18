import torch

from toolbox.states import State, Trackable

if torch.__version__ == '0.4.0':
    from torch.utils.data.sampler import RandomSampler
else:
    from torch.utils.data import RandomSampler


class TrackedRandomSampler(Trackable, RandomSampler):
    def __init__(self, *args, **kwargs):
        super().__init__()
        RandomSampler.__init__(self, *args, **kwargs)

        self.curr_iter = None

        def dump_fn(_):
            return list(self.curr_iter)

        self.left_over = State([], dump_fn=dump_fn)

    def __iter__(self):
        if len(self.left_over) > 0:
            self.curr_iter = iter(self.left_over)
            self.left_over = []
        else:
            self.curr_iter = RandomSampler.__iter__(self)
        return self.curr_iter

    def __len__(self):
        left_over_len = len(self.left_over)
        return (
            left_over_len if left_over_len > 0 else RandomSampler.__len__(self))
