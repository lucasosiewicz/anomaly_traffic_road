from visualization_functions import draw_loss_curves, draw_historgram_of_errors, draw_confusion_matrix
from callbacks.PrintMetricsCallback import PrintMetricsCallback
from counting_functions import count_best_threshold
from data_classes.datamodule import DataModule
from models.ConvAE.ConvAE import ConvAE

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer
import mlflow
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
CALLBACKS = [PrintMetricsCallback(), 
             EarlyStopping(monitor='val_loss', patience=7, mode='min'), 
             ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='src/models/ConvAE/checkpoints/', filename='ConvAE-gray-{epoch}-{val_loss:.2f}')]
LOGGER = MLFlowLogger(experiment_name='DoTA-dataset', run_name='ConvAE-Grayscale', tracking_uri='http://localhost:5000', log_model=True)

def main():
    path_to_data = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DoTA"

    data_module = DataModule(path_to_data)
    data_module.setup()

    model = ConvAE(input_shape=1)
    trainer = Trainer(max_epochs=EPOCHS, accelerator=DEVICE, logger=LOGGER, callbacks=CALLBACKS)
    
    with mlflow.start_run(run_id=LOGGER.run_id):
        trainer.fit(model, data_module)
        trainer.test(model, data_module.test_dataloader())

        train_loss = CALLBACKS[0].train_metrics['loss']
        val_loss = CALLBACKS[0].val_metrics['loss']
        
        draw_loss_curves(train_loss, val_loss, save_path='src/plots')
        draw_historgram_of_errors(model.reconstruction_error, model.targets, save_path='src/plots')
        
        mlflow.log_artifact('src/plots/loss_curves.png')
        mlflow.log_artifact('src/plots/histogram_of_errors.png')

        best_threshold, acc, precision, recall, f1 = count_best_threshold(model.reconstruction_error, model.targets)

        draw_confusion_matrix(model.reconstruction_error, model.targets, best_threshold, save_path='src/plots')
        mlflow.log_artifact('src/plots/confusion_matrix.png')

        mlflow.log_param('best_threshold', best_threshold.item())
        mlflow.log_metrics({'accuracy': acc, 
                            'precision': precision, 
                            'recall': recall, 
                            'f1': f1})


if __name__ == '__main__':
    main()