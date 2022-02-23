import torchvision.models as models
import torch.nn as nn 
import torch 

'''
class Encoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(Encoder, self).__init__()
    self.backbone = models.efficientnet_b2(pretrained=pre_trained)
    self.Block1 = self.backbone.features[:2]
    self.Block2 = self.backbone.features[2]
    self.Block3 = self.backbone.features[3]
    self.Block4 = self.backbone.features[4:6]
    self.Block5 = self.backbone.features[6:8]
  def forward(self,x):
    skip_connections = []
    x  = self.Block1(x)
    skip_connections.append(x)
    x  = self.Block2(x)
    skip_connections.append(x)
    x  = self.Block3(x)
    skip_connections.append(x)
    x  = self.Block4(x)
    skip_connections.append(x)
    x  = self.Block5(x)
    skip_connections = skip_connections[::-1]
    return x,skip_connections 
class Decoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(Decoder, self).__init__()
    self.Encoder = Encoder(pre_trained)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)  
    
    self.Decoder_Block_1 = nn.ConvTranspose2d(352,136,kernel_size=2,stride=2) 
    self.Decoder_Block_2 = nn.ConvTranspose2d(256,80,kernel_size=2,stride=2)
    self.Decoder_Block_3 = nn.ConvTranspose2d(128,40,kernel_size=2,stride=2)
    self.Decoder_Block_4 = nn.ConvTranspose2d(64,16,kernel_size=2,stride=2)
    self.Decoder_Block_5 = nn.ConvTranspose2d(32,16,kernel_size=2,stride=2)
    
    self.Decoder_Block_6 = nn.Conv2d(16, 1, 3, padding=1)
  def forward(self,x): 
    x, skip_connections  = self.Encoder(x)
    
    x = self.Decoder_Block_1(x)
    x = torch.cat([skip_connections[0],x],1) 
    x = self.Decoder_Block_2(x)
    x = torch.cat([skip_connections[1],x],1) 
    x = self.Decoder_Block_3(x)
    x = torch.cat([skip_connections[2],x],1)
    x = self.Decoder_Block_4(x) 
    x = torch.cat([skip_connections[3],x],1)
    x = self.Decoder_Block_5(x)
    x = self.Decoder_Block_6(x)
    return x


'''

class Encoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(Encoder, self).__init__()
    self.backbone = models.efficientnet_b2(pretrained=pre_trained)
    self.Block1 = self.backbone.features[:2]
    self.Block2 = self.backbone.features[2]
    self.Block3 = self.backbone.features[3]
    self.Block4 = self.backbone.features[4:6]
    self.Block5 = self.backbone.features[6:8]
  def forward(self,x):
    skip_connections = []
    x  = self.Block1(x)
    skip_connections.append(x)
    x  = self.Block2(x)
    skip_connections.append(x)
    x  = self.Block3(x)
    skip_connections.append(x)
    x  = self.Block4(x)
    skip_connections.append(x)
    x  = self.Block5(x)
    skip_connections = skip_connections[::-1]
    return x,skip_connections 
from math import ceil
class CNNBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups=1,bias=False):
    super(CNNBlock, self).__init__()  
    self.cnn = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,groups=groups,bias=bias)
    self.bn = nn.BatchNorm2d(out_channels)
    self.silu = nn.SiLU()
  def forward(self,x):
    return self.silu(self.bn(self.cnn(x)))
class SqueezeExcitation(nn.Module):
    def __init__(self,in_channels,reduction_ratio=0.25):
      super(SqueezeExcitation, self).__init__()  
      reduced_dim = int(in_channels*reduction_ratio)  
      self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels,reduced_dim,kernel_size=1,stride=1),
                    nn.SiLU(),
                    nn.Conv2d(reduced_dim,in_channels,kernel_size=1,stride=1),
                    nn.Sigmoid())
    def forward(self,x):
      return x * self.se(x)
class MBCBlock(nn.Module):
  #def __init__(self,input_dim,expansion_ratio,output_dim,stride,kernel_size,padding,reduction_ratio=0.25,survival_prob=0.8,training=True,bias=False):
  def __init__(self,in_channels,out_channels,kernel_size,stride,padding,expand_ratio,reduction_ratio=0.25,survival_prob=0.8):
    super(MBCBlock,self).__init__()
    self.survival_prob = 0.8
    
    self.use_residual = (in_channels == out_channels and stride == 1)
    hidden_dim = in_channels * expand_ratio
    self.expand = (in_channels!=hidden_dim)
    if (self.expand):
      self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
    self.conv2 = CNNBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding,groups=hidden_dim)
    self.SE = SqueezeExcitation(hidden_dim,reduction_ratio)
    self.conv3 = nn.Sequential(nn.Conv2d(hidden_dim, out_channels,kernel_size=1,bias=False),nn.BatchNorm2d(num_features=out_channels))
  def stochastic_depth(self, x): 
    if not self.training:
      return x
    binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
    return torch.div(x, self.survival_prob) * binary_tensor
  def forward(self, x):
    out = self.expand_conv(x) if self.expand else x
    out =  self.conv2(out) 
    out =  self.SE(out)    
    out =  self.conv3(out) 
    if self.use_residual:
      out = self.stochastic_depth(out)
      return out+x
    else:
      return out
class Decoder(nn.Module):
  def __init__(self,pre_trained=False):
    super(Decoder, self).__init__()
    self.Encoder = Encoder(pre_trained)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)  
    self.Decoder_Block_1_1 = MBCBlock(352,136,kernel_size=3, stride = 1,padding=3//2,expand_ratio=3)
    self.Decoder_Block_1_2 = MBCBlock(136,136,kernel_size=3, stride = 1,padding=3//2,expand_ratio=3)
    self.Decoder_Block_2_1 = MBCBlock(256,80,kernel_size=3, stride = 1,padding=3//2,expand_ratio=3)
    self.Decoder_Block_2_2 = MBCBlock(80,80,kernel_size=3, stride = 1,padding=3//2,expand_ratio=3)
    self.Decoder_Block_3_1 = MBCBlock(128,40,kernel_size=3, stride = 1,padding=3//2,expand_ratio=4)
    self.Decoder_Block_3_2 = MBCBlock(40,40,kernel_size=3, stride = 1,padding=3//2,expand_ratio=4)
    self.Decoder_Block_4_1 = MBCBlock(64,16,kernel_size=3, stride = 1,padding=3//2,expand_ratio=5)
    self.Decoder_Block_4_2 = MBCBlock(16,16,kernel_size=3, stride = 1,padding=3//2,expand_ratio=5)
    self.Decoder_Block_5_1 = MBCBlock(32,16,kernel_size=3, stride = 1,padding=3//2,expand_ratio=6) 
    self.Decoder_Block_5_2 = MBCBlock(16,1,kernel_size=3, stride = 1,padding=3//2,expand_ratio=6) 
    
    
    #self.Decoder_Block_6 = nn.Conv2d(16, 1, 3, padding=1)
  def forward(self,x): 
    x, skip_connections  = self.Encoder(x)
    x = self.upsample(x)
    x = self.Decoder_Block_1_1(x)
    x = self.Decoder_Block_1_2(x)
    x = torch.cat([skip_connections[0],x],1) 
    x = self.upsample(x)  
    x = self.Decoder_Block_2_1(x)
    x = self.Decoder_Block_2_2(x)
    x = torch.cat([skip_connections[1],x],1) 
    x = self.upsample(x)
    x = self.Decoder_Block_3_1(x)
    x = self.Decoder_Block_3_2(x)
    x = torch.cat([skip_connections[2],x],1)
    x = self.upsample(x)
    x = self.Decoder_Block_4_1(x)
    x = self.Decoder_Block_4_2(x)  
    x = torch.cat([skip_connections[3],x],1)
    x = self.upsample(x)
    x = self.Decoder_Block_5_1(x)
    x = self.Decoder_Block_5_2(x)
    #x = self.Decoder_Block_6(x)
    return x 

def load_Efficient_UNET(Training=False):
    if Training:
        model = Decoder(pre_trained=True)
    else:
        model = Decoder()    
    return model