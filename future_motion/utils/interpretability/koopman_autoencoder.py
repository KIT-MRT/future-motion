import time
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from einops import rearrange


class BidirectionalKoopmanAE(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        latent_dim,
        learning_rate=1e-3,
        recon_coeff: float = 1.0,
        fwd_coeff: float = 1.0,
        bwd_coeff: float = 0.1,
        con_coeff: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.w_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(input_dim, latent_dim, dtype=torch.float32)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(latent_dim, dtype=torch.float32))
        self.w_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(latent_dim, input_dim, dtype=torch.float32)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))

        self.w_koopman_fwd = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(latent_dim, latent_dim * latent_dim, dtype=torch.float32)
            )
        )
        self.w_koopman_bwd = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(latent_dim, latent_dim * latent_dim, dtype=torch.float32)
            )
        )
        self.i = torch.eye(latent_dim, dtype=torch.float32)

    def encode(self, x, return_koopman_operators: bool = False):
        h = torch.matmul(x, self.w_enc) + self.b_enc
        h = torch.tanh(h)

        c = torch.matmul(h, self.w_koopman_fwd)
        c = rearrange(
            c,
            "... (d0 d1) -> ... d0 d1",
            d0=self.hparams.latent_dim,
            d1=self.hparams.latent_dim,
        )
        d = torch.matmul(h, self.w_koopman_bwd)
        d = rearrange(
            d,
            "... (d0 d1) -> ... d0 d1",
            d0=self.hparams.latent_dim,
            d1=self.hparams.latent_dim,
        )

        if return_koopman_operators:
            return h, c, d

        return h

    def decode(self, x):
        return torch.matmul(x, self.w_dec) + self.b_dec

    def forward(self, x):
        c = self.encode(x)
        x_recon = self.decode(c)
        return x_recon

    def training_step(self, batch, batch_idx):
        # Should work without fusing information in temporal dim
        # since each embedding contains temporal information (speed, temporal index, etc.)
        h, c, d = self.encode(batch, return_koopman_operators=True)
        x_recon = self.decode(h)
        recon_loss = F.mse_loss(x_recon, batch)

        h_past = h[:, :, :-1, None]
        h_future = h[:, :, 1:, None]
        c = c[:, :, :-1]
        d = d[:, :, 1:]

        # Koopman losses
        fwd_pred = torch.matmul(h_past, c)[:, :, :, 0]
        fwd_pred = self.decode(fwd_pred)
        koopman_fwd_loss = F.mse_loss(fwd_pred, batch[:, :, 1:])

        bwd_pred = torch.matmul(h_future, d)[:, :, :, 0]
        bwd_pred = self.decode(bwd_pred)
        koopman_bwd_loss = F.mse_loss(bwd_pred, batch[:, :, :-1])

        koopman_con_loss = 0.5 * torch.norm(
            torch.matmul(c, d) - self.i.to(device=self.device)
        ) + 0.5 * torch.norm(torch.matmul(d, c) - self.i.to(device=self.device))

        loss = (
            self.hparams.recon_coeff * recon_loss
            + self.hparams.fwd_coeff * koopman_fwd_loss
            + self.hparams.bwd_coeff * koopman_bwd_loss
            + self.hparams.con_coeff * koopman_con_loss
        )

        self.log("recon_loss", recon_loss)
        self.log("koopman_fwd_loss", koopman_fwd_loss)
        self.log("koopman_bwd_loss", koopman_bwd_loss)
        self.log("koopman_con_loss", koopman_con_loss)
        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
