"""Training model"""

import logging
from dataclasses import dataclass

import torch
from main import Params
from torch.utils.data import DataLoader

from src.models.lstmvae import *
from src.process_data.make_dataset import MidiDataset
from src.train.encoder_decoder_pretrain import pretrain_autoencoder
from src.train.vae_train import vae_train


@dataclass
class History:
    pretrain_recon: list[float]
    recon: list[float]
    kl: list[float]
    train_loss: list[float]
    test_loss: list[float]
    z: list[torch.Tensor]
    label: list[torch.Tensor]


class Train:
    def __init__(self, params: Params) -> None:
        self.params = params
        self.encoder = Encoder(params=params)
        self.decoder = Decoder(params=params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.95,
        )
        self.history = History(
            pretrain_recon=[],
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder.to(device)
        self.decoder.to(device)

        pretrain_epoch = 20
        self.history.pretrain_recon = pretrain_autoencoder(
            encoder=self.encoder,
            decoder=self.decoder,
            traindata=train_data,
            testdata=test_data,
            epochs=pretrain_epoch,
            optimizer=self.optimizer,
        )

        vae_train(
            model=LSTMVAE(device=device, params=self.params),
            traindata=train_data,
            testdata=test_data,
            epochs=self.params.epoch,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            history=self.history,
        )

        self.history.z = (
            torch.stack(self.history.z)
            .clone()
            .detach()
            .view(-1, self.latent_size)
            .to("cpu")
            .numpy()
        )
        self.history.label = (
            torch.stack(self.history.label).clone().detach().view(-1).to("cpu").numpy()
        )
