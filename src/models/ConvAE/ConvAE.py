import torch
from torch import nn
import lightning as L
from torch import optim


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv_c1 = nn.Conv2d(input_shape, 512, 15, stride=4)
        self.bn_c1 = nn.BatchNorm2d(512)
        self.relu_c1 = nn.ReLU(inplace=True)
        self.pool_c1 = nn.MaxPool2d(2)

        self.conv_c2 = nn.Conv2d(512, 256, 4, stride=1)
        self.bn_c2 = nn.BatchNorm2d(256)
        self.relu_c2 = nn.ReLU(inplace=True)
        self.pool_c2 = nn.MaxPool2d(2)

        self.conv_c3 = nn.Conv2d(256, 128, 3, stride=1)
        self.bn_c3= nn.BatchNorm2d(128)
        self.relu_c3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu_c1(self.bn_c1(self.conv_c1(x)))
        x = self.pool_c1(x)
        x = self.relu_c2(self.bn_c2(self.conv_c2(x)))
        x = self.pool_c2(x)
        x = self.relu_c3(self.bn_c3(self.conv_c3(x)))

        return x

class Decoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.deconv_d1 = nn.ConvTranspose2d(512, input_shape, 15, stride=4)

        self.deconv_d2 = nn.ConvTranspose2d(256, 512, 4, stride=1)
        self.bn_d2_1 = nn.BatchNorm2d(512)
        self.relu_d2 = nn.ReLU(inplace=True)
        self.uppool_d2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn_d2_2 = nn.BatchNorm2d(512)

        self.deconv_d3 = nn.ConvTranspose2d(128, 256, 3, stride=1)
        self.bn_d3_1 = nn.BatchNorm2d(256)
        self.relu_d3 = nn.ReLU(inplace=True)
        self.uppool_d3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn_d3_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.relu_d3(self.bn_d3_1(self.deconv_d3(x)))
        x = self.bn_d3_2(self.uppool_d3(x))
        x = self.relu_d2(self.bn_d2_1(self.deconv_d2(x)))
        x = self.bn_d2_2(self.uppool_d2(x))
        x = self.deconv_d1(x)

        return x
    

class ConvAE(L.LightningModule):
    def __init__(self, encoder=Encoder, decoder=Decoder, input_shape=3, lr=1e-3):
        super().__init__()

        torch.set_float32_matmul_precision('high')

        self.encoder = encoder(input_shape)
        self.decoder = decoder(input_shape)

        self.lr = lr
        self.criterion = nn.MSELoss()

        self.reconstruction_error = None
        self.targets = None

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        batch_errors = []
        for xx, xx_hat in zip(x, x_hat):
            loss = self.criterion(xx, xx_hat)
            batch_errors.append(loss)

        if self.reconstruction_error is None:
            self.reconstruction_error = torch.tensor(batch_errors)
        else:
            self.reconstruction_error = torch.cat([self.reconstruction_error, torch.tensor(batch_errors)])

        if self.targets is None:
            self.targets = y
        else:
            self.targets = torch.cat([self.targets, y])

        self.log("test_loss", torch.mean(torch.tensor(self.reconstruction_error)))

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
