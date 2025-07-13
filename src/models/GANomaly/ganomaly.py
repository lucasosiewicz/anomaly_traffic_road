import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F

# --------------------------------------------------
# GANomaly Lightning Module
# --------------------------------------------------
# Implements encoder-decoder-encoder generator and discriminator
# as described in "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training"
# Samet Akcay et al., CVPR 2019.
# --------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            # For 256x256: 256->128->64->32 natural output
        )
        self.fc = nn.Linear(256 * 32 * 32, latent_dim)

    def forward(self, x):
        feat = self.conv(x)
        flat = feat.view(feat.size(0), -1)
        z = self.fc(flat)
        return z

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=100):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 32 * 32)
        self.deconv = nn.Sequential(
            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 32, 32)
        x_hat = self.deconv(x)
        return x_hat

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # For 256x256: 256->128->64->32 natural output
        self.final = nn.Conv2d(256, 1, 32, 1, 0, bias=False)

    def forward(self, x):
        feats = []
        out = self.conv1(x)
        feats.append(out)
        out = self.conv2(out)
        feats.append(out)
        out = self.conv3(out)
        feats.append(out)
        logit = self.final(out).view(-1)
        return feats, logit

class GANomaly(pl.LightningModule):
    def __init__(self,
                 in_channels=3,
                 latent_dim=100,
                 lr=2e-4,
                 b1=0.5,
                 b2=0.999,
                 w_adv=1.0,
                 w_con=50.0,
                 w_enc=1.0):
        super().__init__()
        self.save_hyperparameters()
        torch.set_float32_matmul_precision('medium')
        # Enable manual optimization for multiple optimizers
        self.automatic_optimization = False
        # Networks
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        self.encoder2 = Encoder(in_channels, latent_dim)
        self.discriminator = Discriminator(in_channels)
        # Loss
        self.bce = nn.BCEWithLogitsLoss()
        
        # Containers for test results (consistent with other models)
        self.reconstruction_error = []
        self.targets = []

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.hparams.lr,
                                 betas=(self.hparams.b1, self.hparams.b2))
        gen_params = list(self.encoder.parameters()) + \
                     list(self.decoder.parameters()) + \
                     list(self.encoder2.parameters())
        opt_g = torch.optim.Adam(gen_params,
                                 lr=self.hparams.lr,
                                 betas=(self.hparams.b1, self.hparams.b2))
        return [opt_d, opt_g], []

    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt_d, opt_g = self.optimizers()
        
        # Generator forward
        z = self.encoder(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        
        # Discriminator updates
        opt_d.zero_grad()
        feats_real, logit_real = self.discriminator(x)
        feats_fake, logit_fake = self.discriminator(x_hat.detach())
        valid = torch.ones_like(logit_real)
        fake = torch.zeros_like(logit_fake)
        loss_real = self.bce(logit_real, valid)
        loss_fake = self.bce(logit_fake, fake)
        loss_d = (loss_real + loss_fake) * 0.5
        self.manual_backward(loss_d)
        opt_d.step()
        self.log('loss_d', loss_d, prog_bar=True)
        
        # Generator updates
        opt_g.zero_grad()
        # Adversarial loss (feature matching)
        feats_real, _ = self.discriminator(x)
        feats_fake, _ = self.discriminator(x_hat)
        Ladv = 0
        for fr, ff in zip(feats_real, feats_fake):
            Ladv += F.mse_loss(ff, fr)
        # Contextual loss
        Lcon = F.l1_loss(x_hat, x)
        # Encoder loss
        Lenc = F.mse_loss(z_hat, z)
        # Total generator loss
        loss_g = self.hparams.w_adv * Ladv + \
                 self.hparams.w_con * Lcon + \
                 self.hparams.w_enc * Lenc
        self.manual_backward(loss_g)
        opt_g.step()
        self.log('train_loss', loss_g, on_epoch=True, prog_bar=False)
        self.log('Ladv', Ladv, prog_bar=False)
        self.log('Lcon', Lcon, prog_bar=False)
        self.log('Lenc', Lenc, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        # anomaly score
        score = torch.mean(torch.abs(z - z_hat), dim=1)
        # Log validation loss for callback compatibility
        val_loss = score.mean()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return score

    def on_validation_epoch_end(self):
        # Updated method name (validation_epoch_end is deprecated)
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        # Calculate anomaly score per sample
        sample_scores = torch.mean(torch.abs(z - z_hat), dim=1)
        
        self.reconstruction_error.extend(sample_scores)
        self.targets.extend(y)

    def on_test_epoch_start(self):
        self.reconstruction_error = []
        self.targets = []

    def on_test_epoch_end(self):
        reconstruction_error = torch.cat(self.reconstruction_error)
        all_targets = torch.cat(self.targets)
        
        self.reconstruction_error = reconstruction_error
        self.targets = all_targets

        self.log("test_loss", reconstruction_error.mean(), on_epoch=True, sync_dist=True)
