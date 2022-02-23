import torch
from torch import nn, Tensor
from torchvision.models import resnet18
from functools import partial

'''
class Encoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(Encoder, self).__init__()
    self.backbone = resnet18( pretrained = pre_trained)
    self.Block_1 = nn.Sequential(*[self.backbone.conv1,self.backbone.bn1,self.backbone.relu])
    self.Block_2 = nn.Sequential(*[self.backbone.maxpool,self.backbone.layer1])
    self.Block_3 = self.backbone.layer2
    self.Block_4 = self.backbone.layer3
    self.Block_5 = self.backbone.layer4
  def forward(self,x):
    skip_connections = []
    x = self.Block_1(x)
    skip_connections.append(x)
    x = self.Block_2(x)
    skip_connections.append(x)
    x = self.Block_3(x)
    skip_connections.append(x)
    x = self.Block_4(x)
    skip_connections.append(x)
    x = self.Block_5(x)
    skip_connections = skip_connections[::-1]
    return x, skip_connections
class Basic_Block(nn.Module):
  expansion = 1
  def __init__(self,input_dim, output_dim,stride=1,bias=False):
    super(Basic_Block, self).__init__()
    self.input_dim=input_dim
    self.output_dim=output_dim
    self.p1 = nn.Sequential(nn.Conv2d(input_dim,output_dim,kernel_size=3,stride=stride,padding=1, bias=bias),nn.BatchNorm2d(output_dim),nn.ReLU(True))
    self.p2 = nn.Sequential(nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=1,padding=1, bias=bias),nn.BatchNorm2d(output_dim))
    self.downsample = nn.Sequential()
    if (input_dim != output_dim or stride!=1):
      self.downsample = nn.Sequential(nn.Conv2d(input_dim,output_dim,kernel_size=1, stride=stride, bias=bias),nn.BatchNorm2d(output_dim))
    self.ReLU = nn.ReLU(True)   
  def forward(self,x):
    out = self.p1(x)
    out = self.p2(out)   
    out = out + self.downsample(x) 
    out = self.ReLU(out)
    return out    
class ResNet18_UNet(nn.Module):
  def __init__(self,pre_trained=False): 
    super(ResNet18_UNet, self).__init__()
    self.Encoder = Encoder(pre_trained = pre_trained)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)  
    self.Decoder_Block_1 = nn.Sequential(Basic_Block(768,256),Basic_Block(256,256))
    self.Decoder_Block_2 = nn.Sequential(Basic_Block(384,128),Basic_Block(128,128)) 
    self.Decoder_Block_3 = nn.Sequential(Basic_Block(192,64),Basic_Block(64,64)) 
    self.Decoder_Block_4 = nn.Sequential(Basic_Block(128,32),Basic_Block(32,32)) 
    self.ConvTranspose5 = nn.ConvTranspose2d(32, 1, kernel_size=4, padding=1, stride=2)
  def forward(self,x):
    x , skip_connection = self.Encoder(x)
    x = self.upsample(x)
    x = torch.cat([skip_connection[0],x],1)
    x = self.Decoder_Block_1(x)
    x = self.upsample(x)
    x = torch.cat([skip_connection[1],x],1)
    x = self.Decoder_Block_2(x)
    x = self.upsample(x)
    x = torch.cat([skip_connection[2],x],1)
    x = self.Decoder_Block_3(x)
    x = self.upsample(x)
    x = torch.cat([skip_connection[3],x],1)
    x = self.Decoder_Block_4(x)
    x = self.ConvTranspose5(x) 
    return x

def load_ResNet18_UNET(Training=False):
    if Training:
        model = ResNet18_UNet(pre_trained=True)
    else:
        model = ResNet18_UNet()    
    return model     
    
'''

import torch
from torch import nn, Tensor
from torchvision.models import resnet18
from functools import partial


class Encoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(Encoder, self).__init__()
    self.backbone = resnet18( pretrained = pre_trained)
    self.Block_1 = nn.Sequential(*[self.backbone.conv1,self.backbone.bn1,self.backbone.relu])
    self.Block_2 = nn.Sequential(*[self.backbone.maxpool,self.backbone.layer1])
    self.Block_3 = self.backbone.layer2
    self.Block_4 = self.backbone.layer3
    self.Block_5 = self.backbone.layer4
  def forward(self,x):
    skip_connections = []
    x = self.Block_1(x)
    skip_connections.append(x)
    x = self.Block_2(x)
    skip_connections.append(x)
    x = self.Block_3(x)
    skip_connections.append(x)
    x = self.Block_4(x)
    skip_connections.append(x)
    x = self.Block_5(x)
    skip_connections = skip_connections[::-1]
    return x, skip_connections
   
class ResNet18_UNet(nn.Module):
  def __init__(self,pre_trained=False): 
    super(ResNet18_UNet, self).__init__()
    self.Encoder = Encoder(pre_trained = pre_trained)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False) 
    self.Decoder_Block_1 = nn.ConvTranspose2d(512,512,kernel_size=2,stride=2) 
    self.Decoder_Block_2 = nn.ConvTranspose2d(768,256,kernel_size=2,stride=2)
    self.Decoder_Block_3 = nn.ConvTranspose2d(384,128,kernel_size=2,stride=2)
    self.Decoder_Block_4 = nn.ConvTranspose2d(192,64,kernel_size=2,stride=2)
    self.Decoder_Block_5 = nn.ConvTranspose2d(128,32,kernel_size=2,stride=2)
    self.Conv6 = nn.Conv2d(32,1,kernel_size=3,padding=1, bias=False)
    
    
  def forward(self,x):
    x , skip_connection = self.Encoder(x)
    x = self.Decoder_Block_1(x)
    x = torch.cat([skip_connection[0],x],1)
    x = self.Decoder_Block_2(x)
    x = torch.cat([skip_connection[1],x],1)
    x = self.Decoder_Block_3(x)
    x = torch.cat([skip_connection[2],x],1)
    x = self.Decoder_Block_4(x)
    x = torch.cat([skip_connection[3],x],1)
    x = self.Decoder_Block_5(x)
    x = self.Conv6(x) 
    return x

def load_ResNet18_UNET(Training=False):
    if Training:
        model = ResNet18_UNet(pre_trained=True)
    else:
        model = ResNet18_UNet()    
    return model     