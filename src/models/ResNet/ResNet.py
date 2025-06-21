import torch
from torch import nn
import lightning as pl
from torchmetrics import Accuracy
import torchvision.models as models


class ResNet(pl.LightningModule):
    def __init__(self, input_shape=1, num_classes=2, learning_rate=0.0005, transform=None, freeze=True, class_weights=[1] * 6, weight_decay=0):
        super(ResNet, self).__init__()
        self.save_hyperparameters()

        torch.set_float32_matmul_precision('high')

        self.transform = transform
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.class_weights = class_weights
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.model = models.resnet18(weights='DEFAULT')

        # Freeze pre-trained layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes if num_classes > 2 else 1)
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=self.model.conv1.bias is not None
        )

        # Define a loss function and metric
        if len(self.class_weights) > 2:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Define metric
        if self.num_classes == 2:
            self.accuracy = Accuracy(task="binary")
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # Container for predictions
        self.reconstruction_error = []
        # Container for targets
        self.targets = []

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.transform:
            x = self.transform(x)
        return x, y

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        
        # For binary classification with BCELoss, squeeze the output to match target shape
        if self.num_classes == 2:
            outputs = outputs.squeeze()
            labels = labels.float()  # BCELoss expects float targets
        
        loss = self.criterion(outputs, labels)

        # Calculate and log accuracy
        if self.num_classes == 2:
            predicted_classes = (torch.sigmoid(outputs) > 0.5).long()
        else:
            predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels.long())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        
        # For binary classification with BCELoss, squeeze the output to match target shape
        if self.num_classes == 2:
            outputs = outputs.squeeze()
            labels = labels.float()  # BCELoss expects float targets
        
        loss = self.criterion(outputs, labels)

        if self.num_classes == 2:
            predicted_classes = (torch.sigmoid(outputs) > 0.5).long()
        else:
            predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels.long())
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        
        # For binary classification with BCELoss, squeeze the output to match target shape
        if self.num_classes == 2:
            outputs = outputs.squeeze()
            labels = labels.float()  # BCELoss expects float targets
        
        loss = self.criterion(outputs, labels)

        if self.num_classes == 2:
            predicted_classes = (torch.sigmoid(outputs) > 0.5).long()
            # For binary classification, create 2D probabilities [prob_class_0, prob_class_1]
            prob_class_1 = torch.sigmoid(outputs)
            prob_class_0 = 1 - prob_class_1
            predicted_probs = torch.stack([prob_class_0, prob_class_1], dim=1)
        else:
            predicted_classes = torch.argmax(outputs, dim=1)
            predicted_probs = torch.softmax(outputs, dim=1)
            
        acc = self.accuracy(predicted_classes, labels.long())
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        # Save predictions and targets for later use
        self.reconstruction_error.append(predicted_probs)
        self.targets.append(labels.long())

    def on_test_epoch_end(self):
        all_losses = torch.cat(self.reconstruction_error).cpu()
        all_targets = torch.cat(self.targets).cpu()
        
        self.reconstruction_error = all_losses
        self.targets = all_targets

        self.log("test_loss", all_losses.mean(), on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {'optimizer': optimizer}