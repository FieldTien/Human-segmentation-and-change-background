import torch 
import torch.nn as nn 

import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self,in_channels,reduction_ratio=.25):
      super(SqueezeExcitation, self).__init__()  
      reduced_dim = round(in_channels*reduction_ratio)  
      self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels,reduced_dim,kernel_size=1,stride=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduced_dim,in_channels,kernel_size=1,stride=1),
                    nn.Hardsigmoid(inplace=True))
    def forward(self,x):
      out = self.se(x)
      return out*x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, kernal_size, expand_channels, out_channels, uese_SE, use_HS,stride,bias = False):
        super(InvertedResidual, self).__init__()
        padding = (kernal_size - 1) // 2
        if use_HS == "HS" :
          activation = nn.Hardswish(inplace=True)
        elif use_HS == "RE" :
          activation = nn.ReLU(inplace=True)
        self.use_connect = stride == 1 and in_channels == out_channels  
        if(in_channels!=expand_channels):
          self.conv1 = nn.Sequential(nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, bias=bias),nn.BatchNorm2d(num_features=expand_channels),activation)
        else:
          self.conv1 = nn.Sequential()
        self.conv2 = nn.Sequential(nn.Conv2d(expand_channels, expand_channels, kernel_size=kernal_size, stride=stride, padding=padding ,groups = expand_channels, bias=bias),nn.BatchNorm2d(num_features=expand_channels),activation)
        
        if(uese_SE):
          self.squeeze_block = SqueezeExcitation(expand_channels)
        else:
          self.squeeze_block = nn.Sequential() 

        self.conv3 = nn.Sequential(nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, bias=bias),nn.BatchNorm2d(num_features=out_channels))
    
    def forward(self,x):
        out =  self.conv1(x)
        out =  self.conv2(out) 
        out =  self.squeeze_block(out)  
        out =  self.conv3(out)   
        out =  out + x if self.use_connect==1 else out 
        return out
from torchvision.models import mobilenet_v3_large
class MobileNetV3_Encoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(MobileNetV3_Encoder, self).__init__()
    self.features= mobilenet_v3_large(pretrained=pre_trained).features
  def forward(self,x):
    skip_connections = []
    for index in range(len(self.features)):
      if(index > 1 and index < 16):
        if(self.features[index].block[1][0].stride[0] == 2):
          skip_connections.append(x)
      x = self.features[index](x) 
    skip_connections = skip_connections[::-1]    
    return x,skip_connections
'''
class DecoderBlock(nn.Module):
    def __init__(self, Inverter_Block1, Inverter_Block2):
        super(DecoderBlock, self).__init__()
        self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False) 
        self.Inverter_Block1 = Inverter_Block1
        self.Inverter_Block2 = Inverter_Block2
    def forward(self,x,skip_connection):
        x = self.upsample(x)
        x = self.Inverter_Block1(x)
        x = torch.cat([skip_connection,x],1)
        x = self.Inverter_Block2(x)
        return x
class UNET_MobileV3(nn.Module):
    def __init__(self,out_channels=1,load_pretrain=False):
        super(UNET_MobileV3, self).__init__()
        if(load_pretrain):
          self.Encoder = MobileNetV3_Encoder(pre_trained=True)
        else:
          self.Encoder = MobileNetV3_Encoder()
        channel_1 = 112
        self.Inverter_Block1_1 = InvertedResidual(960, 5, 960 ,channel_1 ,True, "HS", 1)
        self.Inverter_Block1_2 = InvertedResidual(channel_1*2, 5, channel_1*6 ,channel_1 ,True, "HS", 1)
        self.DecoderBlock1 = DecoderBlock(self.Inverter_Block1_1,self.Inverter_Block1_2)

        channel_2 = 40
        self.Inverter_Block2_1 = InvertedResidual(channel_1, 3, channel_1*4 ,channel_2 ,False, "HS", 1)
        self.Inverter_Block2_2 = InvertedResidual(channel_2*2, 3, channel_2*4 ,channel_2 ,False, "HS", 1)
        self.DecoderBlock2 = DecoderBlock(self.Inverter_Block2_1,self.Inverter_Block2_2)

        channel_3 = 24
        self.Inverter_Block3_1 = InvertedResidual(channel_2, 5, channel_2*3 ,channel_3 ,True, "RE", 1)
        self.Inverter_Block3_2 = InvertedResidual(channel_3*2, 5, channel_3*3 ,channel_3 ,True, "RE", 1)
        self.DecoderBlock3 = DecoderBlock(self.Inverter_Block3_1,self.Inverter_Block3_2)

        channel_4 = 16
        self.Inverter_Block4_1 = InvertedResidual(channel_3, 3, channel_3*3 ,channel_4 ,False, "RE", 1)
        self.Inverter_Block4_2 = InvertedResidual(channel_4*2, 3, channel_4*3 ,channel_4 ,False, "RE", 1)
        self.DecoderBlock4 = DecoderBlock(self.Inverter_Block4_1,self.Inverter_Block4_2)

        self.ConvTranspose5 = nn.ConvTranspose2d(channel_4, out_channels, kernel_size=4, padding=1, stride=2)
    def forward(self,x):
        x, skip_connection = self.Encoder(x) 
        x = self.DecoderBlock1(x,skip_connection[0])
        x = self.DecoderBlock2(x,skip_connection[1])
        x = self.DecoderBlock3(x,skip_connection[2])
        x = self.DecoderBlock4(x,skip_connection[3])
        x = self.ConvTranspose5(x)
        return x  

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
class UNET_MobileV3(nn.Module):
    def __init__(self,out_channels=1,load_pretrain=False):
        super(UNET_MobileV3, self).__init__()
        if(load_pretrain):
          self.Encoder = MobileNetV3_Encoder(pre_trained=True)
        else:
          self.Encoder = MobileNetV3_Encoder()
        channel_1 = 112
        self.Inverter_Block1 = InvertedResidual(channel_1*2, 5, channel_1*6 ,channel_1 ,True, "HS", 1)
        self.DecoderBlock1 = DecoderBlock(960, channel_1, self.Inverter_Block1)

        channel_2 = 40
        self.Inverter_Block2 = InvertedResidual(channel_2*2, 3, channel_2*6 ,channel_2 ,True, "HS", 1)
        self.DecoderBlock2 = DecoderBlock(channel_1, channel_2, self.Inverter_Block2)

        channel_3 = 24
        self.Inverter_Block3 = InvertedResidual(channel_3*2, 5, channel_3*6 ,channel_3 ,True, "RE", 1)
        self.DecoderBlock3 = DecoderBlock(channel_2, channel_3, self.Inverter_Block3)

        channel_4 = 16
        self.Inverter_Block4 = InvertedResidual(channel_4*2, 3, channel_4*6 ,channel_4 ,True, "RE", 1)
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
'''
class SqueezeExcitation(nn.Module):
    def __init__(self,in_channels,reduction_ratio=.25):
      super(SqueezeExcitation, self).__init__()  
      reduced_dim = round(in_channels*reduction_ratio)  
      self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels,reduced_dim,kernel_size=1,stride=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduced_dim,in_channels,kernel_size=1,stride=1),
                    nn.Hardsigmoid(inplace=True))
    def forward(self,x):
      out = self.se(x)
      return out*x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, kernal_size, expand_channels, out_channels, uese_SE, use_HS,stride,bias = False):
        super(InvertedResidual, self).__init__()
        padding = (kernal_size - 1) // 2
        if use_HS == "HS" :
          activation = nn.Hardswish(inplace=True)
        elif use_HS == "RE" :
          activation = nn.ReLU(inplace=True)
        self.use_connect = stride == 1 and in_channels == out_channels  
        if(in_channels!=expand_channels):
          self.conv1 = nn.Sequential(nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, bias=bias),nn.BatchNorm2d(num_features=expand_channels),activation)
        else:
          self.conv1 = nn.Sequential()
        self.conv2 = nn.Sequential(nn.Conv2d(expand_channels, expand_channels, kernel_size=kernal_size, stride=stride, padding=padding ,groups = expand_channels, bias=bias),nn.BatchNorm2d(num_features=expand_channels),activation)
        
        if(uese_SE):
          self.squeeze_block = SqueezeExcitation(expand_channels)
        else:
          self.squeeze_block = nn.Sequential() 

        self.conv3 = nn.Sequential(nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, bias=bias),nn.BatchNorm2d(num_features=out_channels))
    
    def forward(self,x):
        out =  self.conv1(x)
        out =  self.conv2(out) 
        out =  self.squeeze_block(out)  
        out =  self.conv3(out)   
        out =  out + x if self.use_connect==1 else out 
        return out
from torchvision.models import mobilenet_v3_large
class MobileNetV3_Encoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(MobileNetV3_Encoder, self).__init__()
    self.features= mobilenet_v3_large(pretrained=pre_trained).features
  def forward(self,x):
    skip_connections = []
    for index in range(len(self.features)):
      if(index > 1 and index < 16):
        if(self.features[index].block[1][0].stride[0] == 2):
          skip_connections.append(x)
      x = self.features[index](x) 
    skip_connections = skip_connections[::-1]    
    return x,skip_connections
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Inverter_Block1,Inverter_Block2):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2)
        self.Inverter_Block1 = Inverter_Block1
        self.Inverter_Block2 = Inverter_Block2
    def forward(self,x,skip_connection):
        x = self.deconv(x)
        x = torch.cat([skip_connection,x],1)
        x = self.Inverter_Block1(x)
        x = self.Inverter_Block2(x)
        return x
class UNET_MobileV3(nn.Module):
    def __init__(self,out_channels=1,load_pretrain=False):
        super(UNET_MobileV3, self).__init__()
        if(load_pretrain):
          self.Encoder = MobileNetV3_Encoder(pre_trained=True)
        else:
          self.Encoder = MobileNetV3_Encoder()
        channel_1 = 112
        self.Inverter_Block1_1 = InvertedResidual(channel_1*2, 5, channel_1*6 ,channel_1 ,False, "HS", 1)
        self.Inverter_Block1_2 = InvertedResidual(channel_1, 5, channel_1*6 ,channel_1 ,False, "HS", 1)

        self.DecoderBlock1 = DecoderBlock(960, channel_1, self.Inverter_Block1_1, self.Inverter_Block1_2)

        channel_2 = 40
        self.Inverter_Block2_1 = InvertedResidual(channel_2*2, 3, channel_2*6 ,channel_2 ,True, "HS", 1)
        self.Inverter_Block2_2 = InvertedResidual(channel_2, 3, channel_2*6 ,channel_2 ,True, "HS", 1)

        self.DecoderBlock2 = DecoderBlock(channel_1, channel_2, self.Inverter_Block2_1,self.Inverter_Block2_2)

        channel_3 = 24
        self.Inverter_Block3_1 = InvertedResidual(channel_3*2, 5, channel_3*6 ,channel_3 ,False, "RE", 1)
        self.Inverter_Block3_2 = InvertedResidual(channel_3, 5, channel_3*6 ,channel_3 ,False, "RE", 1)

        self.DecoderBlock3 = DecoderBlock(channel_2, channel_3, self.Inverter_Block3_1,self.Inverter_Block3_2 )

        channel_4 = 16
        self.Inverter_Block4_1 = InvertedResidual(channel_4*2, 3, channel_4*6 ,channel_4 ,True, "RE", 1)
        self.Inverter_Block4_2 = InvertedResidual(channel_4, 3, channel_4*6 ,channel_4 ,True, "RE", 1)

        self.DecoderBlock4 = DecoderBlock(channel_3, channel_4, self.Inverter_Block4_1,self.Inverter_Block4_2 )

        self.ConvTranspose5 = nn.ConvTranspose2d(channel_4, out_channels, kernel_size=4, padding=1, stride=2)
    def forward(self,x):
        x, skip_connection = self.Encoder(x) 
        x = self.DecoderBlock1(x,skip_connection[0])
        x = self.DecoderBlock2(x,skip_connection[1])
        x = self.DecoderBlock3(x,skip_connection[2])
        x = self.DecoderBlock4(x,skip_connection[3])
        x = self.ConvTranspose5(x)
        return x
                  
def load_MobileV3_UNET(Training=False):
    
    if Training:
       model = UNET_MobileV3(load_pretrain=True)
    else:
       model = UNET_MobileV3()    
    return model



if __name__ == "__main__":
   print(load_MobileV3_UNET(Training=False)(torch.randn(1,3,256,256)).shape)    