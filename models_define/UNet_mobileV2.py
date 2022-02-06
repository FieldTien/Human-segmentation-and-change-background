import torch

from torch import nn
from torch import Tensor
from typing import Callable, List, Optional


from .mobile_utils import ConvNormActivation, _make_divisible


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_Encoder(nn.Module):
    def __init__(
        self,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2_Encoder, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer,
                                                        activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        skip_connections = []
        for index in range(len(self.features)):
          if(index > 1 and index < 18):
            if(self.features[index].conv[1][0].stride[0] == 2): 
              skip_connections.append(x)
          x = self.features[index](x)    
        skip_connections = skip_connections[::-1]  

        return x , skip_connections


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
 

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Inverter_Block):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2)
        self.Inverter_Block = Inverter_Block
    def forward(self,x,skip_connection):
        x = self.deconv(x)
        x = torch.cat([skip_connection,x],1)
        x = self.Inverter_Block(x)
        return x
    
class UNET_MobileV2(nn.Module):
    def __init__(self,out_channels=1,load_pretrain=False):
        super(UNET_MobileV2, self).__init__()
        expand_ratio = 6
        self.Encoder = MobileNetV2_Encoder()
        if(load_pretrain):
           import torchvision.models as models
           pretrain = models.mobilenet_v2(pretrained=True)
           self.Encoder.features.load_state_dict(pretrain.features.state_dict())
        channel_1 = 96
        self.Inverter_Block1 = InvertedResidual(channel_1*2, channel_1, stride=1, expand_ratio = expand_ratio)
        self.DecoderBlock1 = DecoderBlock(1280, channel_1, self.Inverter_Block1)

        channel_2 = 32
        self.Inverter_Block2 = InvertedResidual(channel_2*2, channel_2, stride=1, expand_ratio = expand_ratio)
        self.DecoderBlock2 = DecoderBlock(channel_1, channel_2, self.Inverter_Block2)

        channel_3 = 24
        self.Inverter_Block3 = InvertedResidual(channel_3*2, channel_3, stride=1, expand_ratio = expand_ratio)
        self.DecoderBlock3 = DecoderBlock(channel_2, channel_3, self.Inverter_Block3)

        channel_4 = 16
        self.Inverter_Block4 = InvertedResidual(channel_4*2, channel_4, stride=1, expand_ratio = expand_ratio)
        self.DecoderBlock4 = DecoderBlock(channel_3, channel_4, self.Inverter_Block4)

        self.ConvTranspose5 = nn.ConvTranspose2d(channel_4, out_channels, kernel_size=4, padding=1, stride=2)
    def forward(self,x):
        x, skip_connection = self.Encoder(x) 
        x = self.DecoderBlock1(x,skip_connection[0])
        x = self.DecoderBlock2(x,skip_connection[1])
        x = self.DecoderBlock3(x,skip_connection[2])
        x = self.DecoderBlock4(x,skip_connection[3])
        x = self.ConvTranspose5(x)
        return x     
    
def load_MobileV2_UNET(Training=False):
    if Training:
       model = UNET_MobileV2(load_pretrain=True)
    else:
       model = UNET_MobileV2()    
    return model    
  


    