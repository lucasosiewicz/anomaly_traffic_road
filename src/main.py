from visualization_functions import draw_loss_curves, draw_historgram_of_errors, draw_confusion_matrix
from callbacks.PrintMetricsCallback import PrintMetricsCallback
from counting_functions import count_best_threshold
from data_classes.datamodule import DataModule
from models.model_factory import get_model

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer
import mlflow
import torch
import yaml  # Import YAML library

# Function to load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()

    # Use values from config
    path_to_data = config['data']['path']
    batch_size = config['data']['batch_size']
    sequence_length = config['data']['sequence_length']
    stride = config['data']['stride']
    model_name = config['model']['name']
    epochs = config['trainer']['epochs']
    device = config['trainer']['device']
    experiment_name = config['logging']['experiment_name']
    run_name = config['logging']['run_name']
    tracking_uri = config['logging']['tracking_uri']
    log_model_flag = config['logging']['log_model']
    checkpoint_dir = config['logging']['checkpoint_dir']
    checkpoint_filename = config['logging']['checkpoint_filename']
    early_stopping_patience = config['callbacks']['early_stopping_patience']

    # Determine device
    if device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        resolved_device = device

    # Get model and related flags
    model, is_unsupervised, is_sequence = get_model(model_name)

    # Setup Callbacks
    callbacks = [
        PrintMetricsCallback(),
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min'),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            dirpath=checkpoint_dir,
            filename=checkpoint_filename
        )
    ]

    # Setup Logger
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        log_model=log_model_flag
    )

    # Setup DataModule
    data_module = DataModule(
        path_to_data=path_to_data,
        batch_size=batch_size,
        unsupervised=is_unsupervised,
        is_sequence=is_sequence,
        sequence_length=sequence_length,
        stride=stride
    )
    data_module.setup()

    # Setup Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=resolved_device,
        logger=logger,
        callbacks=callbacks
    )

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_params(config)
        trainer.fit(model, data_module)
        trainer.test(model, data_module.test_dataloader(), ckpt_path='best')

        # Pobieranie metryk
        train_loss = callbacks[0].train_metrics['loss']
        val_loss = callbacks[0].val_metrics['loss']
        
        # Wizualizacja wyników
        draw_loss_curves(train_loss, val_loss, save_path='src/plots')
        mlflow.log_artifact('src/plots/loss_curves.png')
        
        if is_unsupervised:
            draw_historgram_of_errors(model.reconstruction_error, model.targets, save_path='src/plots')
            mlflow.log_artifact('src/plots/histogram_of_errors.png')

        # Obliczanie najlepszego progu i metryk
        best_threshold, acc, precision, recall, f1 = count_best_threshold(
            model.reconstruction_error,
            model.targets,
            unsupervised=is_unsupervised
        )

        # Wizualizacja i logowanie macierzy pomyłek
        draw_confusion_matrix(
            model.reconstruction_error,
            model.targets,
            save_path='src/plots',
            unsupervised=is_unsupervised
        )
        mlflow.log_artifact('src/plots/confusion_matrix.png')

        # Logowanie metryk
        mlflow.log_param('best_threshold', best_threshold.item() if best_threshold is not None else None)
        mlflow.log_metrics({
            'accuracy': acc, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1
        })


if __name__ == '__main__':
    main()