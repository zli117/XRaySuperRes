import torch
from torch import nn
from torch.utils.data import DataLoader

from toolbox.states import Trackable, TorchState, State, save_on_interrupt
from .progress_bar import ProgressBar


def save_model(model: nn.Module, file_path, epoch, seed, step=None):
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'seed': seed,
                'step': step}, file_path)


class TrackedTraining(Trackable):
    def __init__(self, model: nn.Module, train_dataset, valid_dataset,
                 optimizer_cls, model_save_path_prefix,
                 state_save_path_prefix, optimizer_config: dict,
                 train_loader_config: dict, inference_loader_config: dict,
                 epochs=1, gpu=True, progress_bar_size=20, save_optimizer=True):
        self.model = TorchState(model)
        self.optimizer_cls = optimizer_cls
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            **optimizer_config)
        self.optimizer = TorchState(optimizer) if save_optimizer else optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # Avoid saving sampler twice
        self.train_sampler = train_loader_config.pop('sampler', None)

        self.model_save_path = model_save_path_prefix
        self.state_save_path = state_save_path_prefix
        self.epochs = epochs
        self.curr_epochs = State(0)
        self.curr_steps = State(0)
        self.train_loader_config = State(train_loader_config)
        self.inference_loader_config = State(inference_loader_config)
        self.gpu = gpu
        self.progress_bar_size = State(progress_bar_size)

    def parse_train_batch(self, batch):
        return torch.Tensor(0.0), torch.Tensor(0.0)

    def parse_valid_batch(self, batch):
        return self.parse_train_batch(batch)

    def loss_fn(self, output, target):
        return torch.Tensor(0.0)

    def validate(self, data_loader):
        total_data = 0
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                ipt, target = self.parse_valid_batch(batch)
                output = self.model(ipt)
                batch_size = output.shape[0]
                loss = self.loss_fn(output, target)
                total_loss += loss * batch_size
                total_data += batch_size
        return total_loss / total_data

    def train(self):
        @save_on_interrupt(self.state_save_path + 'interrupt.state')
        def _train(self):
            if torch.cuda.is_available() and self.gpu:
                self.model.cuda()

                # Manually moving optimizer state to GPU
                # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                torch.cuda.empty_cache()

            train_loader = DataLoader(self.train_dataset,
                                      sampler=self.train_sampler,
                                      **self.train_loader_config)
            valid_loader = DataLoader(self.valid_dataset,
                                      **self.inference_loader_config)

            while self.curr_epochs < self.epochs:
                self.model.train()
                total_steps = self.curr_steps + len(train_loader)
                progress_bar = ProgressBar(self.progress_bar_size,
                                           ' loss: %.06f, batch: %d, epoch: %d')
                for _, batch in enumerate(train_loader):
                    ipt, target = self.parse_train_batch(batch)
                    output = self.model(ipt)
                    loss = self.loss_fn(output, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    progress_bar.progress(
                        self.curr_steps / total_steps * 100,
                        loss, self.curr_steps, self.curr_epochs)
                    self.curr_steps += 1
                self.curr_epochs += 1
                self.curr_steps = 0

                save_model(self.model,
                           '%s_%d.model' % (
                               self.model_save_path, self.curr_epochs),
                           self.curr_epochs, 0)
                self.save_state(
                    save_path='%s_%d.state' % (
                        self.state_save_path, self.curr_epochs))
                validate_loss = self.validate(valid_loader)
                print('\nValidation loss: %f' % validate_loss)

            return self.model

        return _train(self)


class DummyTrainClass(TrackedTraining):
    def __init__(self, model):
        super().__init__(model, None, None, lambda *args: None, None, None, {},
                         {}, {})
