import torch
from torch import nn
import lightning as L
from torch import optim
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=3, padding=1, stride=stride)
            for (in_, out_), stride in zip([[input_shape, 32], [32, 64], [64, 64]], [1,2,2])
        ])
        self.normalization_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=channels)
            for channels in [32, 64, 64]
        ])
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        for conv_layer, normalization_layer in zip(self.conv_layers, self.normalization_layers):
            x = self.activation(normalization_layer(conv_layer(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, output_shape):
        super().__init__()

        self.conv_transpose_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_, out_channels=out_, kernel_size=3, padding=1, stride=stride, output_padding=(1 if stride == 2 else 0))
            for (in_, out_), stride in zip([[64, 64], [64, 64], [64, 32]], [1,2,2])
        ])
        self.normalization_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=channels)
            for channels in [64, 64, 32, output_shape]
        ])
        self.activation = nn.LeakyReLU()
        self.last_layer = nn.ConvTranspose2d(in_channels=32, out_channels=output_shape, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x):
        for conv_transpose_layer, normalization_layer in zip(self.conv_transpose_layers, self.normalization_layers):
            x = self.activation(normalization_layer(conv_transpose_layer(x)))
            #print(x.shape)
        return self.last_layer(x)


class LargerConvAE(L.LightningModule):
    def __init__(self, encoder=Encoder, decoder=Decoder, input_shape=1, learning_rate=0.001):
        super().__init__()

        torch.set_float32_matmul_precision('high')

        self.encoder = encoder(input_shape)
        self.decoder = decoder(input_shape)

        self.learning_rate = learning_rate
        self.criterion = SSIM(reduction='none')

        self.reconstruction_error = []
        self.targets = []

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = 1 - self.criterion(x_hat, x)
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, sync_dist=True)
        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = 1 - self.criterion(x_hat, x)
        self.log("val_loss", loss.mean(), on_step=True, on_epoch=True, sync_dist=True)
        return loss.mean()

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = 1 - self.criterion(x_hat, x)
        sample_loss = loss.view(loss.size(0), -1).mean(dim=1)
        
        self.reconstruction_error.append(sample_loss)
        self.targets.append(y)
    
    def on_test_epoch_end(self):
        all_losses = torch.cat(self.reconstruction_error).cpu()
        all_targets = torch.cat(self.targets).cpu()
        
        self.reconstruction_error = all_losses
        self.targets = all_targets

        self.log("test_loss", all_losses.mean(), on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer}
    
