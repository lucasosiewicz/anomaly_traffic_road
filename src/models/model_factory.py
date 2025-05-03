from torch import nn

from .ConvAE.ConvAE import ConvAE
from .ResNet.ResNet import ResNet
from .ResNetLSTM.ResNetLSTM import ResNetLSTM
from .LargerConvAE.LargerConvAE import LargerConvAE

def get_model(model_name: str, learning_rate: float) -> tuple[nn.Module, bool, bool]:
    if model_name == 'ConvAE':
        return ConvAE(learning_rate=learning_rate), True, False
    elif model_name == 'ResNet':
        return ResNet(learning_rate=learning_rate), False, False
    elif model_name == 'ResNetLSTM':
        return ResNetLSTM(learning_rate=learning_rate), False, True
    elif model_name == 'LargerConvAE':
        return LargerConvAE(learning_rate=learning_rate), True, False
    else:
        raise ValueError(f"Model {model_name} not found")

