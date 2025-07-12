from torch import nn

from .ConvAE.ConvAE import ConvAE
from .ResNet.ResNet import ResNet
from .ResNetLSTM.ResNetLSTM import ResNetLSTM
from .LargerConvAE.LargerConvAE import LargerConvAE
from .FineGrained.fine_grained import FineGrainedModel
from .GANomaly.ganomaly import GANomaly

def get_model(model_name: str, learning_rate: float, class_weights: list[float], weight_decay: float, input_channels: int = 1, num_classes: int = 6) -> tuple[nn.Module, bool, bool]:
    if model_name == 'ConvAE':
        return ConvAE(learning_rate=learning_rate, input_shape=input_channels), True, False
    elif model_name == 'ResNet':
        return ResNet(
            learning_rate=learning_rate, 
            class_weights=class_weights, 
            weight_decay=weight_decay, 
            input_shape=input_channels,
            num_classes=num_classes
        ), False, False
    elif model_name == 'ResNetLSTM':
        return ResNetLSTM(learning_rate=learning_rate), False, True
    elif model_name == 'LargerConvAE':
        return LargerConvAE(learning_rate=learning_rate), True, False
    elif model_name == 'FineGrained':
        return FineGrainedModel(
            num_classes=num_classes,
            learning_rate=learning_rate,
            class_weights=class_weights,
            weight_decay=weight_decay
        ), False, False
    elif model_name == 'GANomaly':
        return GANomaly(
            in_channels=input_channels,
            lr=learning_rate
        ), True, False
    else:
        raise ValueError(f"Model {model_name} not found")

