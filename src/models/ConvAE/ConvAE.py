import torch
from torch import nn
import lightning as L
from torch import optim


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 512, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=5, padding=2)
        self.deconv4 = nn.ConvTranspose2d(512, input_shape[0], kernel_size=11, padding=5)

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.deconv4(x)

        return x
    

class ConvAE(L.LightningModule):
    def __init__(self, encoder, decoder, input_shape, lr, transform=None):
        super().__init__()
        self.encoder = encoder(input_shape)
        self.decoder = decoder(input_shape)

        self.lr = lr

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer