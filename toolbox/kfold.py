from random import randint

from sklearn.model_selection import KFold

from toolbox.states import Trackable, State, save_on_interrupt
from toolbox.train import TrackedTraining, DummyTrainClass


class TrackedKFold(Trackable):
    def __init__(self, state_save_path, model, k_folds, total_data, groups=None,
                 shuffle=True):
        self.state_save_path = state_save_path
        # Won't save model twice
        self.model = model
        self.total_data = State(total_data)
        self.groups = State(groups)
        self.k_folds = State(k_folds)
        self.shuffle = State(shuffle)
        self.k_fold_seed = State(randint(0, 1e7))
        self.fold_idx = State(0)
        self.train_obj = DummyTrainClass()
        self.results = State([])

    def get_train_obj(self, train_idx) -> TrackedTraining:
        pass

    def test(self, test_idx):
        pass

    def run(self):
        @save_on_interrupt(self.state_save_path + 'interrupt.state')
        def _run(self):
            k_fold = KFold(self.k_folds, random_state=self.k_fold_seed,
                           shuffle=self.shuffle)
            for i, train_test in enumerate(
                    k_fold.split(self.data_x, self.data_y)):
                if i < self.fold_idx:
                    continue
                self.train_obj = self.get_train_obj(train_test[0])
                print('Fold:', i)
                self.train_obj.train()
                test_result = self.test(train_test[1])
                self.results.append(test_result)
                print('Fold %d test result: %s' % (i, test_result))

        _run(self)
