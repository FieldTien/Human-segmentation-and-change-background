
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
from torchvision.models import vgg16_bn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, 3 ,1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, 3 ,1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
         return self.conv(x)   



class Encoder(nn.Module): 
     def __init__(self, load_pretrain):   
         super(Encoder, self).__init__()
         modules= vgg16_bn(pretrained=load_pretrain).features               
         self.BLOCK_1 = modules[:6]        
         self.BLOCK_2 = modules[6:13]
         self.BLOCK_3 = modules[13:23]
         self.BLOCK_4 = modules[23:33]
         self.pool = modules[33:40]
 
     def forward(self,x):
         skip_connections = [] 
         x = self.BLOCK_1(x)   
         skip_connections.append(x)
         x = self.BLOCK_2(x)  
         skip_connections.append(x) 
         x = self.BLOCK_3(x)  
         skip_connections.append(x) 
         x = self.BLOCK_4(x)  
         skip_connections.append(x) 
         x = self.pool(x)  
         skip_connections = skip_connections[::-1]
         return x, skip_connections
         
         
class UNET_VGG16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features = [64,128,256,512] , load_pretrain=False):
        super(UNET_VGG16, self).__init__()
       
        self.downs = Encoder(load_pretrain=load_pretrain)
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
            self.ups.append(DoubleConv(feature*2,feature))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1],features[-1]*2, 3 ,1, 1, bias=False),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            
        )
        self.fianl_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        x, skip_connections = self.downs(x)
        x = self.bottleneck(x)
        
        for index in range(0,len(self.ups),2):
            x = self.ups[index](x)
            skip_connection =skip_connections[index//2]
            if x.shape[2:] != skip_connection.shape[2:]:
                x = TF.resize(x,size=skip_connection.shape[2:])
            concat_skip_connect = torch.cat((skip_connection,x),dim=1)
            x = self.ups[index+1](concat_skip_connect)
        return self.fianl_conv(x)
def load_VGG16_UNET(Training=False):
    if Training:
       model = UNET_VGG16(load_pretrain=True)
    else:
       model = UNET_VGG16()    
    return model  


if __name__ == "__main__":
    model = UNET_VGG16()
    x = torch.randn([1,3,256,256])   
    y = model(x)
    print(y.shape)
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()
    print('Number of Weight %d' %(params))    
    #modules= vgg16_bn(pretrained=True).features
    #print(modules[:40])



