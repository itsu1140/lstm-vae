import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.lstmvae import LSTMVAE
from src.train.train import History


def vae_train(
    model: LSTMVAE,
    traindata: DataLoader,
    testdata: DataLoader,
    epochs: int,
    optimizer,
    scheduler,
    history: History,
    beta_max: float = 1.0,
    warmup_epochs: int = 20,
):
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()
        train_loss: float = 0
        epoch_kl: float = 0
        epoch_recon: float = 0
        for data, labels in traindata:
            output, recon_loss, kl_loss, z = model(data, epochs)
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            epoch_recon += recon_loss
            epoch_kl += kl_loss
            if epochs - epoch < epochs / 10:
                history.z.append(z)
                history.label.append(labels)

            log: str = (
                f"[VAE] epoch={epoch + 1} "
                f"loss={loss.item():.4f} "
                f"recon={recon_loss.item():.4f} "
                f"kl={kl_loss.item():.4f} "
            )
            logger.info(log)
        history.train_loss.append(train_loss)
        history.recon.append(epoch_recon)
        history.kl.append(epoch_kl)

        model.eval()
        test_loss: float = 0
        with torch.no_grad():
            for data, labels in testdata:
                _, recon_loss, kl_loss, _ = model(data, epochs)
                test_loss += recon_loss + kl_loss
        history.test_loss.append(test_loss)
        scheduler.step()
