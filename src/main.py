from data_classes.datamodule import DataModule
from models.ConvAE.ConvAE import ConvAE

from lightning import Trainer


def main():

    # ========================== DATA MODULE ========================== #
    data_module = DataModule()
    data_module.setup()

    # ========================== MODEL ========================== #
    model = ConvAE()

    # ========================== TRAINING ========================== #
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data_module)

    # ========================== TESTING ========================== #
    trainer.test(model, data_module.test_dataloader())


if __name__ == '__main__':
    main()