from visualization_functions import (
    draw_loss_curves, 
    draw_historgram_of_errors, 
    draw_confusion_matrix, 
    visualize_autoencoder_reconstructions, 
    draw_histogram_of_errors_by_class
)
from callbacks.PrintMetricsCallback import PrintMetricsCallback
from counting_functions import count_best_threshold, measure_single_sample_inference_time
from data_classes.datamodule import DataModule
from models.model_factory import get_model

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer
import mlflow
import torch
import yaml  # Import YAML library
import time # Dodajemy import time


# TODO:
# - undersampling of major class - NOT YET
# - crop images to objects - NOT YET
# - check how reconstructed images look like - DONE
# - check distribution of errors of each class on historgram - DONE
# - experiment with diffrent images sizes - NOT YET
# - save most affordable model (training time, inference time, size) - NOT YET

# Function to load config
def load_config(config_path='src/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()

    # Use values from config
    path_to_data = config['data']['path']
    dataset = config['data']['dataset']
    batch_size = config['data']['batch_size']
    sequence_length = config['data']['sequence_length']
    stride = config['data']['stride']
    crop_type = config['data']['crop_type']
    target_class = config['data']['target_class']
    ego_involved = config['data']['ego_involved']
    color_space = config['data']['color_space']
    model_name = config['model']['name']
    class_weights = config['model']['class_weights']
    weight_decay = config['model']['weight_decay']
    epochs = config['trainer']['epochs']
    device = config['trainer']['device']
    learning_rate = config['trainer']['learning_rate']
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
    model, is_unsupervised, is_sequence = get_model(model_name, learning_rate, class_weights, weight_decay)

    # Obliczanie liczby parametrów modelu
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())

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
        dataset=dataset,
        batch_size=batch_size,
        unsupervised=is_unsupervised,
        is_sequence=is_sequence,
        sequence_length=sequence_length,
        stride=stride,
        crop_type=crop_type,
        target_class=target_class,
        ego_involved=ego_involved,
        color_space=color_space
    )
    data_module.unsupervised = is_unsupervised
    data_module.is_sequence = is_sequence
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
        mlflow.log_artifact('src/config.yaml')
        mlflow.log_param('num_trainable_params', num_trainable_params)
        mlflow.log_param('num_total_params', num_total_params)

        # Trening modelu
        start_train_time = time.time()
        trainer.fit(model, data_module)
        end_train_time = time.time()
        training_duration_seconds = end_train_time - start_train_time
        mlflow.log_metric('training_duration_seconds', training_duration_seconds)

        # Testowanie modelu
        trainer.test(model, data_module.test_dataloader(), ckpt_path='best')

        # Pomiar czasu inferencji dla pojedynczej próbki
        single_sample_inference_duration_ms = measure_single_sample_inference_time(
            model=model,
            dataloader=data_module.test_dataloader(),
            device=resolved_device,
            is_unsupervised=is_unsupervised,
            is_sequence=is_sequence
        )
        if single_sample_inference_duration_ms is not None:
            mlflow.log_metric('single_sample_inference_duration_ms', single_sample_inference_duration_ms)
            
        # Pobieranie metryk
        train_loss = callbacks[0].train_metrics['loss']
        val_loss = callbacks[0].val_metrics['loss']
        
        # Wizualizacja wyników
        draw_loss_curves(train_loss, val_loss, save_path='src/plots')
        mlflow.log_artifact('src/plots/loss_curves.png')
        
        if is_unsupervised:
            draw_historgram_of_errors(model.reconstruction_error, model.targets, save_path='src/plots')
            mlflow.log_artifact('src/plots/histogram_of_errors.png')

            draw_histogram_of_errors_by_class(model.reconstruction_error, model.targets, save_path='src/plots')
            mlflow.log_artifact('src/plots/histogram_of_errors_by_class.png')

            # Wizualizacja rekonstrukcji autoenkodera
            visualize_autoencoder_reconstructions(
                autoencoder=model, 
                dataset=data_module.test_dataset, 
                save_path='src/plots'
            )
            mlflow.log_artifact('src/plots/autoencoder_reconstructions.png')

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
            best_threshold,
            save_path='src/plots',
            unsupervised=is_unsupervised
        )
        mlflow.log_artifact('src/plots/confusion_matrix.png')

        # Logowanie metryk
        mlflow.log_param('best_threshold', best_threshold.item() if best_threshold != 0 else None)
        mlflow.log_metrics({
            'accuracy': acc, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1
        })


if __name__ == '__main__':
    main()