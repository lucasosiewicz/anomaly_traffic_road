import torch
from torch import nn
import lightning as pl
from torchmetrics import Accuracy
import torchvision.models as models


class ResNet(pl.LightningModule):
    def __init__(self, input_shape=1, num_classes=2, learning_rate=0.0005, transform=None, freeze=True, class_weights=None, weight_decay=0):
        super(ResNet, self).__init__()
        self.save_hyperparameters()

        torch.set_float32_matmul_precision('high')

        self.transform = transform
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Set default class_weights if not provided
        if class_weights is None:
            self.class_weights = [1.0] * num_classes
        else:
            self.class_weights = class_weights

        self.model = models.resnet18(weights='DEFAULT')

        # Freeze pre-trained layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        # Always use num_classes for output size
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=input_shape,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=self.model.conv1.bias is not None
        )

        # Define a loss function and metric - use num_classes consistently
        if self.num_classes > 2:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float))
        else:
            # For binary classification, use CrossEntropyLoss with 2 classes
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float))
        
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
        
        loss = self.criterion(outputs, labels)

        # Calculate and log accuracy
        predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        
        loss = self.criterion(outputs, labels)

        predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        
        loss = self.criterion(outputs, labels)

        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_probs = torch.softmax(outputs, dim=1)
            
        acc = self.accuracy(predicted_classes, labels)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        # Save predictions and targets for later use
        self.reconstruction_error.append(predicted_probs)
        self.targets.append(labels)

    def on_test_epoch_start(self):
        self.reconstruction_error = []
        self.targets = []

    def on_test_epoch_end(self):
        all_losses = torch.cat(self.reconstruction_error).cpu()
        all_targets = torch.cat(self.targets).cpu()
        
        self.reconstruction_error = all_losses
        self.targets = all_targets

        self.log("test_loss", all_losses.mean(), on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        params_to_train = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.learning_rate / 10,
            max_lr=self.learning_rate,
            step_size_up=2000,
            mode="triangular"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }