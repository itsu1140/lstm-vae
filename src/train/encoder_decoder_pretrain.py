import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.lstmvae import Decoder, Encoder


def pretrain_autoencoder(
    encoder: Encoder,
    decoder: Decoder,
    traindata: DataLoader,
    testdata: DataLoader,
    epochs: int,
    optimizer,
):
    logger = logging.getLogger(__name__)

    encoder.train()
    decoder.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_list: list[float] = []

    for epoch in range(epochs):
        loss = 0
        for data in traindata:
            data = data.to(device)

            mu, _ = encoder(data)
            seq_len = data.size(1)
            out = decoder(mu, seq_len)
            recon_loss = F.mse_loss(out, data)
            loss += recon_loss.item()

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            log = (
                f"[AE pretrain] epoch: {epoch + 1}, recon loss: {recon_loss.item():.4f}"
            )
            logger.info(log)
        loss_list.append(loss)
    return loss_list
