from torch import nn

from .ConvAE.ConvAE import ConvAE
from .ResNet.ResNet import ResNet
from .ResNetLSTM.ResNetLSTM import ResNetLSTM
from .LargerConvAE.LargerConvAE import LargerConvAE

def get_model(model_name: str) -> tuple[nn.Module, bool, bool]:
    if model_name == 'ConvAE':
        return ConvAE(), True, False
    elif model_name == 'ResNet':
        return ResNet(), False, True
    elif model_name == 'ResNetLSTM':
        return ResNetLSTM(), False, True
    elif model_name == 'LargerConvAE':
        return LargerConvAE(), True, False
    else:
        raise ValueError(f"Model {model_name} not found")

