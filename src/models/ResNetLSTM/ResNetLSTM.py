import torch
from torch import nn
import lightning as pl
from torchmetrics import Accuracy
import torchvision.models as models


class ResNetLSTM(pl.LightningModule):
    def __init__(
        self,
        input_shape=1,
        learning_rate=0.0005,
        transform=None,
        freeze_resnet=True,
        hidden_size=512,
        num_layers=2,
        dropout=0.5
    ):
        super(ResNetLSTM, self).__init__()
        self.save_hyperparameters()

        torch.set_float32_matmul_precision('high')

        self.transform = transform
        self.learning_rate = learning_rate
        self.input_shape = input_shape

        # Inicjalizacja ResNet
        self.resnet = models.resnet18(weights='DEFAULT')
        
        # Freeze pre-trained layers
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Dostosowanie pierwszego warstwy konwolucyjnej
        self.resnet.conv1 = nn.Conv2d(
            in_channels=input_shape,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None
        )

        # Usunięcie ostatniej warstwy fully connected z ResNet
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.resnet.fc.in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Warstwa klasyfikacyjna
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Zmiana na 1 wyjście
            nn.Sigmoid()  # Dodanie sigmoid dla klasyfikacji binarnej
        )

        # Definicja funkcji straty i metryki
        pos_weight = torch.tensor([0.61/0.39])  # Waga dla klasy pozytywnej (1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.accuracy = Accuracy(task="binary")

        # Kontenery na predykcje i etykiety
        self.reconstruction_error = []
        self.targets = []

    def forward(self, x):
        # x ma kształt [batch_size, sequence_length, channels, height, width]
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape dla ResNet
        x = x.view(-1, c, h, w)  # [batch_size * seq_len, channels, height, width]
        
        # Ekstrakcja cech przez ResNet
        features = self.feature_extractor(x)  # [batch_size * seq_len, features]
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, features]
        
        # Przetwarzanie przez LSTM
        lstm_out, _ = self.lstm(features)  # [batch_size, seq_len, hidden_size * 2]
        
        # Przewidywanie dla każdej klatki w sekwencji
        out = self.classifier(lstm_out)  # [batch_size, seq_len, num_classes]
        
        return out

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.transform:
            x = self.transform(x)
        return x, y

    def training_step(self, batch, batch_idx):
        sequences, labels = batch  # labels: [batch_size]
        labels = labels.float()  # Konwersja na float dla BCE
        outputs = self(sequences)  # [batch_size, sequence_length, 1]
        
        # Bierzemy tylko przewidywanie z ostatniej klatki
        outputs = outputs[:, -1, :]  # [batch_size, 1]

        loss = self.criterion(outputs, labels.unsqueeze(1))

        # Obliczanie i logowanie dokładności
        predicted_classes = (outputs > 0.5).float()
        acc = self.accuracy(predicted_classes, labels.unsqueeze(1))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        labels = labels.float()
        outputs = self(sequences)

        # Bierzemy tylko przewidywanie z ostatniej klatki
        outputs = outputs[:, -1, :]

        loss = self.criterion(outputs, labels.unsqueeze(1))
        predicted_classes = (outputs > 0.5).float()
        acc = self.accuracy(predicted_classes, labels.unsqueeze(1))

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        labels = labels.float()
        outputs = self(sequences)

        # Bierzemy tylko przewidywanie z ostatniej klatki
        outputs = outputs[:, -1, :]

        loss = self.criterion(outputs, labels.unsqueeze(1))
        predicted_classes = (outputs > 0.5).float()
        acc = self.accuracy(predicted_classes, labels.unsqueeze(1))

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        predicted_probs = outputs  # Już mamy sigmoid w modelu

        # Zapisywanie predykcji i etykiet
        self.reconstruction_error.append(predicted_probs)
        self.targets.append(labels.unsqueeze(1))

    def on_test_epoch_end(self):
        all_losses = torch.cat(self.reconstruction_error).cpu()
        all_targets = torch.cat(self.targets).cpu()

        self.reconstruction_error = all_losses
        self.targets = all_targets

        self.log("test_loss", all_losses.mean(), on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        trainable_params = list(self.lstm.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.hparams.learning_rate)
        return {'optimizer': optimizer}
