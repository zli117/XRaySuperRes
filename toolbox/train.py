import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from toolbox.misc import save_model, cuda
from toolbox.progress_bar import ProgressBar
from toolbox.states import Trackable, TorchState, State, save_on_interrupt


class TrackedTraining(Trackable):
    def __init__(self, model: nn.Module, train_dataset, valid_dataset,
                 optimizer_cls, save_dir, optimizer_config: dict,
                 train_loader_config: dict, inference_loader_config: dict,
                 epochs=1, gpu=True, progress_bar_size=20, save_optimizer=True):
        super().__init__()
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

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.epochs = epochs
        self.curr_epochs = State(0)
        self.curr_steps = State(0)
        self.train_loader_config = State(train_loader_config)
        self.inference_loader_config = State(inference_loader_config)
        self.gpu = gpu
        self.progress_bar_size = State(progress_bar_size)
        self.valid_loss_history = State([])
        self.train_loss_history = State([])
        self.curr_train_loss = State([])

    def parse_train_batch(self, batch):
        return torch.Tensor(0.0), torch.Tensor(0.0)

    def parse_valid_batch(self, batch):
        return self.parse_train_batch(batch)

    def train_loss_fn(self, output, target):
        return torch.Tensor(0.0)

    def valid_loss_fn(self, output, target):
        return self.train_loss_fn(output, target)

    def validate(self, data_loader):
        total_data = 0
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                ipt, target = self.parse_valid_batch(batch)
                output = self.model(ipt)
                batch_size = output.shape[0]
                loss = self.valid_loss_fn(output, target)
                total_loss += loss * batch_size
                total_data += batch_size
        return total_loss / total_data

    def one_epoch(self, train_loader):
        self.model.train()
        total_steps = self.curr_steps + len(train_loader)
        progress_bar = ProgressBar(self.progress_bar_size,
                                   ' loss: %.06f, batch: %d, epoch: %d')
        losses = []
        for batch in train_loader:
            ipt, target = self.parse_train_batch(batch)
            output = self.model(ipt)
            loss = self.train_loss_fn(output, target)
            losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.curr_train_loss.append(float(loss))
            progress_bar.progress(
                self.curr_steps / total_steps * 100,
                loss, self.curr_steps, self.curr_epochs)
            self.curr_steps += 1
        print('\nAverage train loss: %.06f' % (sum(losses) / len(losses)))
        self.curr_epochs += 1
        self.curr_steps = 0

        self.train_loss_history.append(self.curr_train_loss)
        self.curr_train_loss = []

        save_model(self.model,
                   os.path.join(self.save_dir, '%d.model' % self.curr_epochs),
                   self.curr_epochs, 0)
        self.save_state(save_path=os.path.join(self.save_dir,
                                               '%d.state' % self.curr_epochs))

    @save_on_interrupt(
        lambda self: os.path.join(self.save_dir, 'interrupt.state'))
    def train(self):
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
            self.one_epoch(train_loader)
            validate_loss = self.validate(valid_loader)
            self.valid_loss_history.append(float(validate_loss))
            print('Validation loss: %f' % validate_loss)

        return self.model


class TrackedTrainingGAN(TrackedTraining):
    def __init__(self, d_model: nn.Module, model: nn.Module, train_dataset,
                 valid_dataset, optimizer_cls, save_dir,
                 d_optimizer_config: dict, g_optimizer_config: dict,
                 train_loader_config: dict, inference_loader_config: dict,
                 discriminator_loss, epochs=1, gpu=True, progress_bar_size=20,
                 save_optimizer=True, discriminator_weight=1.0):
        super().__init__(model, train_dataset, valid_dataset, optimizer_cls,
                         save_dir,
                         g_optimizer_config, train_loader_config,
                         inference_loader_config, epochs, gpu,
                         progress_bar_size, save_optimizer)
        self.d_model = TorchState(d_model)
        d_optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, d_model.parameters()),
            **d_optimizer_config)
        self.discriminator_loss = discriminator_loss
        self.d_optimizer = TorchState(
            d_optimizer) if save_optimizer else d_optimizer
        self.discriminator_weight = discriminator_weight

    def train_loss_fn(self, output, target):
        """
        Computes extra loss for training the generator
        Args:
            output: The output of the generator
            target: The real image

        Returns:
            The loss
        """
        return 0

    def one_epoch(self, train_loader):
        real_label = 1
        fake_label = 0

        self.model.train()
        self.d_model.train()
        progress_bar = ProgressBar(self.progress_bar_size,
                                   ' d_loss: %.06f, g_loss: %.06f, epoch: %d')
        total_steps = self.curr_steps + len(train_loader)
        d_losses = []
        g_losses = []

        for batch in train_loader:
            g_ipt, real = self.parse_train_batch(batch)

            # train discriminator with real
            self.d_optimizer.zero_grad()
            batch_size = real.shape[0]
            label = cuda(torch.full((batch_size, 1), real_label))
            output = self.d_model(real)
            loss_real = self.discriminator_loss(output, label)
            loss_real.backward()

            # train discriminator with fake
            fake = self.model(g_ipt)
            label.fill_(fake_label)
            output = self.d_model(fake.detach())
            loss_fake = self.discriminator_loss(output, label)
            loss_fake.backward()
            self.d_optimizer.step()
            loss_discriminator = loss_real + loss_fake
            d_losses.append(loss_discriminator)

            # train generator
            label.fill_(real_label)
            output = self.d_model(fake)
            loss_d = self.discriminator_loss(output, label)
            loss_g = self.discriminator_weight * loss_d + self.train_loss_fn(
                fake, real)
            self.optimizer.zero_grad()
            loss_g.backward()
            self.optimizer.step()
            g_losses.append(loss_g)

            self.curr_steps += 1
            progress_bar.progress(self.curr_steps / total_steps * 100,
                                  loss_discriminator, loss_g, self.curr_epochs)

        print('\nAverage d loss: %.06f, g loss: %.06f' % (
            sum(d_losses) / len(d_losses), sum(g_losses) / len(g_losses)))
        self.curr_epochs += 1
        self.curr_steps = 0

        save_model(self.model,
                   os.path.join(self.save_dir,
                                'g_%d.model' % self.curr_epochs),
                   self.curr_epochs, 0)
        save_model(self.d_model,
                   os.path.join(self.save_dir,
                                'd_%d.model' % self.curr_epochs),
                   self.curr_epochs, 0)
        self.save_state(save_path=os.path.join(self.save_dir,
                                               '%d.state' % self.curr_epochs))

    def train(self):
        if torch.cuda.is_available() and self.gpu:
            self.d_model.cuda()

            # Manually moving optimizer state to GPU
            # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
            for state in self.d_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            torch.cuda.empty_cache()
        return super().train(), self.d_model
