import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from einops import rearrange


class SAE(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        latent_dim,
        max_epochs=10000,
        sparsity_coeff=3e-4,
        learning_rate=1e-3,
    ):
        super().__init__()
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
        self.w_dec.data[:] = self.w_dec / self.w_dec.norm(dim=-1, keepdim=True)
        self.b_dec = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))

        self.save_hyperparameters()

        self.loss = []
        self.l1_loss = []
        self.l2_loss = []
        self.reconst_loss = []

    def encode(self, x):
        x = x - self.b_dec
        s = torch.matmul(x, self.w_enc) + self.b_enc
        s = torch.relu(s)
        return s

    def decode(self, x):
        return torch.matmul(x, self.w_dec) + self.b_dec

    def forward(self, x):
        s = self.encode(x)
        x_recon = self.decode(s)
        return x_recon

    def training_step(self, batch, batch_idx):
        # In an autoencoder the input is the same as labels
        batch = batch[0]

        s = self.encode(batch)
        x_recon = self.decode(s)

        # Sum over latent dim, then mean over batch
        l2_loss = (x_recon - batch).pow(2).sum(-1).mean(0)
        l1_loss = s.abs().sum()

        loss = l2_loss + self.hparams.sparsity_coeff * l1_loss
        loss = loss.mean(dim=0)

        total_recon_loss = (batch - x_recon) ** 2

        self.loss.append(loss)
        self.l1_loss.append(l1_loss.mean(dim=0))
        self.l2_loss.append(l2_loss.mean(dim=0))
        self.reconst_loss.append(total_recon_loss.mean(dim=0))

        # Log the metric for checkpointing
        avg_total_loss = torch.stack(self.loss).mean()
        self.log("loss", avg_total_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def on_train_epoch_end(self) -> None:
        avg_total_loss = torch.stack(self.loss).mean()
        avg_l1_loss = torch.stack(self.l1_loss).mean()
        avg_l2_loss = torch.stack(self.l2_loss).mean()
        avg_reconst_loss = torch.stack(self.reconst_loss).mean()

        # Prepare log messages
        self.log("loss", avg_total_loss, sync_dist=True)
        self.log("l1_loss", avg_l1_loss, sync_dist=True)
        self.log("l2_loss", avg_l2_loss, sync_dist=True)
        self.log("reconstruction_loss", avg_reconst_loss, sync_dist=True)

        # Reset loss trackers for the next epoch
        self.loss = []
        self.l1_loss = []
        self.l2_loss = []
        self.reconst_loss = []


class CunninghamSAE(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, sparsity_coeff=3e-4, learning_rate=1e-3):
        super().__init__()
        self.w = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(input_dim, latent_dim, dtype=torch.float32)
            )
        )
        self.b = nn.Parameter(torch.zeros(latent_dim, dtype=torch.float32))
        self.save_hyperparameters()

    def encode(self, x):
        c = torch.matmul(x, self.w) + self.b
        c = torch.relu(c)
        return c

    def decode(self, x):
        return torch.matmul(x, self.w.T)

    def forward(self, x):
        c = self.encode(x)
        x_recon = self.decode(c)
        return x_recon

    def training_step(self, batch, batch_idx):
        c = self.encode(batch)
        x_recon = self.decode(c)

        recon_loss = F.mse_loss(x_recon, batch)
        sparsity_loss = torch.mean(torch.abs(c))

        loss = recon_loss + self.hparams.sparsity_coeff * sparsity_loss

        self.log("recon_loss", recon_loss)
        self.log("sparsity_loss", sparsity_loss)
        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
