from lightning.pytorch.callbacks import Callback


class PrintMetricsCallback(Callback):
    def __init__(self):
        self.train_metrics = {"loss": []}
        self.val_metrics = {"loss": []}

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss'].item()
        print(f"Train Loss: {train_loss:.4f}")

        self.train_metrics['loss'].append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics['val_loss'].item()
        print(f"Epoch: {trainer.current_epoch}, "
              f"Val Loss: {val_loss:.4f}", end=' || ')

        self.val_metrics['loss'].append(val_loss)

