"""Training model"""

import logging
from dataclasses import dataclass

import torch
from main import Params
from torch.utils.data import DataLoader

from src.models.lstmvae import LSTMVAE
from src.process_data.make_dataset import MidiDataset


@dataclass
class History:
    recon: list[float]
    kl: list[float]
    train_loss: list[float]
    test_loss: list[float]
    z: list[torch.Tensor]
    label: list[torch.Tensor]


class Train:
    def __init__(self, model: LSTMVAE, params: Params) -> None:
        self.params = params
        self.batch_size = params.batch_size
        self.epoch = params.epoch
        self.model = model
        self.latent_size = params.latent_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.95,
        )
        self.history = History(
            recon=[],
            kl=[],
            train_loss=[],
            test_loss=[],
            z=[],
            label=[],
        )
        self.label2genre = {}

    def make_dataset(self) -> tuple[DataLoader, DataLoader]:
        # train : test = 8 : 2
        dataset = MidiDataset(self.params)
        self.label2genre = dataset.get_label2genre()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
        )
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_loader, test_loader

    def train(self) -> None:
        # torch.backends.cudnn.benchmark = True
        train_data, test_data = self.make_dataset()
        logger = logging.getLogger(__name__)

        for epoch in range(self.epoch):
            reconloss = 0
            klloss = 0
            train_loss = 0
            test_loss = 0

            # train
            self.model.train()
            for data, labels, chroma in train_data:
                _, recon, kl, z = self.model.forward(data, epoch, chroma)
                loss = recon + kl
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                reconloss += recon
                klloss += kl
                train_loss += loss
                if self.epoch - epoch < 30:
                    self.history.z.append(z)
                    self.history.label.append(labels)
            train_loss /= self.batch_size
            reconloss /= self.batch_size
            klloss /= self.batch_size
            logger.info(
                f"epoch={epoch + 1}, recon loss={reconloss:.4f}, kl loss={klloss:.4f}, train loss={train_loss:.4f}",
                end=" ",
            )
            self.scheduler.step()
            self.history.train_loss.append(train_loss)
            self.history.recon.append(reconloss)
            self.history.kl.append(klloss)

            # test
            self.model.eval()
            with torch.no_grad():
                for data, labels, chroma in test_data:
                    _, recon, kl, _ = self.model.forward(data, epoch, chroma)
                    loss = recon + kl
                    test_loss += loss
            test_loss /= self.batch_size
            logger(f"test loss={test_loss:.4f}")
            self.history.train_loss.append(test_loss)
        last = -15
        self.history.z = (
            torch.stack(self.history.z[last:])
            .clone()
            .detach()
            .view(-1, self.latent_size)
            .to("cpu")
            .numpy()
        )
        self.history.label = (
            torch.stack(self.history.label[last:])
            .clone()
            .detach()
            .view(-1)
            .to("cpu")
            .numpy()
        )
