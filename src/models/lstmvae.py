"""LSTM-VAE model class"""

import torch
from main import Params
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, params: Params) -> None:
        super().__init__()
        bidirection = True
        self.lstm = nn.LSTM(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            bidirectional=bidirection,
        )
        cell_out = params.hidden_size * params.num_layers * (1 + bidirection)
        self.fc_mu = nn.Linear(cell_out, params.latent_size)
        self.logvar = nn.Linear(cell_out, params.latent_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch_size, sequence_len, hidden_size)"""
        # encode
        batch_size = x.shape[0]
        _, (hidden, _) = self.lstm(x)  # 4 x batch_size x hidden_size
        h_n = (
            hidden.permute(1, 0, 2)  # swap 1dim, 2dim (batch_size x 4 x hidden_size)
            .contiguous()
            .view(batch_size, -1)  # batch_size x (hidden_size * 4)
        )
        # vae
        mu = self.fc_mu(h_n)  # fully connect
        logvar = self.logvar(h_n)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, params: Params) -> None:
        super().__init__()
        self.input_size = params.input_size
        self.hidden_size = params.hidden_size
        biderection = True
        self.lstm = nn.LSTM(
            input_size=params.latent_size,
            hidden_size=self.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            bidirectional=biderection,
        )
        self.vae_h0 = nn.Linear(params.latent_size, params.hidden_size)
        self.vae_c0 = nn.Linear(params.latent_size, params.hidden_size)
        self.vae_fc = nn.Linear(params.latent_size, params.input_size * params.seq_len)
        self.dec_fc = nn.Linear(
            params.hidden_size * (1 + biderection),
            params.output_size,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        z: torch.Tensor,
        seq_len: int,
        genre_z: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_z = genre_z if genre_z is not None else z
        h0 = (
            self.vae_h0(hidden_z).unsqueeze(0).repeat(4, 1, 1).contiguous()
        )  # num_layers x batch x hidden
        c0 = self.vae_c0(hidden_z).unsqueeze(0).repeat(4, 1, 1).contiguous()
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(z, (h0, c0))

        fc = self.dec_fc(output)
        fc = torch.sigmoid(fc)
        return fc


class LSTMVAE(nn.Module):
    def __init__(
        self,
        device: torch.device,
        params: Params,
    ) -> None:
        super().__init__()
        self.output_size = params.output_size
        self.device = device
        self.seq_len = params.seq_len
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def reparameterize(self, mu: torch.Tensor, log: torch.Tensor) -> torch.Tensor:
        ep = torch.randn_like(mu).to(self.device)
        return mu + torch.exp(log * 0.5) * ep

    def forward(
        self,
        x: torch.Tensor,
        epoch: int,
        chroma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        chroma = chroma.to(self.device)

        # LSTM encoder
        mu, logvar = self.encoder(x)

        # VAE
        z = self.reparameterize(mu, logvar)  # batch x latent

        # LSTM decoder
        output = self.decoder(z, self.seq_len)

        # calculate loss
        recon, kl = self.loss_function(
            pred=output,
            target=chroma,
            mu=mu,
            logvar=logvar,
            epoch=epoch,
        )

        return output, recon, kl, z

    def loss_function(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # recon = F.smooth_l1_loss(input=pred, target=target, reduction="mean")
        recon = F.binary_cross_entropy(pred, target, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        kl = torch.mean(kl)
        kl_weight = min(1, (epoch + 1) / 500) * 100
        return recon, kl * kl_weight
