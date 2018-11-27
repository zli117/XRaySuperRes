from random import randint

import numpy as np
from sklearn.model_selection import KFold

from toolbox.states import Trackable, State, save_on_interrupt
from toolbox.train import TrackedTraining


class TrackedKFold(Trackable):
    def __init__(self, state_save_dir, model, k_folds, data_length,
                 groups=None, shuffle=True):
        super().__init__()
        self.state_save_dir = state_save_dir
        # Won't save model twice
        self.model = model
        self.data_length = State(data_length)
        self.groups = State(groups)
        self.k_folds = State(k_folds)
        self.shuffle = State(shuffle)
        self.k_fold_seed = State(randint(0, 1e7))
        self.fold_idx = State(0)
        # TODO: Find a better way to create initial train obj
        self.train_obj = self.get_train_obj(np.arange(data_length))
        self.results = State([])

    @property
    def history(self):
        return self.results

    def get_train_obj(self, train_idx) -> TrackedTraining:
        pass

    def test(self, test_idx):
        pass

    @save_on_interrupt(lambda self: self.state_save_path + 'interrupt.state')
    def run(self):
        k_fold = KFold(self.k_folds, random_state=self.k_fold_seed,
                       shuffle=self.shuffle)
        dummy_x = np.arange(self.data_length).reshape(-1, 1)
        dummy_y = np.arange(self.data_length)
        for i, train_test in enumerate(k_fold.split(dummy_x, dummy_y)):
            if i < self.fold_idx:
                continue
            if not self.restored:
                self.train_obj = self.get_train_obj(train_test[0])
            print('Fold:', self.fold_idx)
            self.train_obj.train()
            test_result = self.test(train_test[1])
            self.results.append(test_result)
            print('Fold %d test result: %s' % (self.fold_idx, test_result))
            self.fold_idx += 1
