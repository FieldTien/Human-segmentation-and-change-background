import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import HumanDataset

def save_checkpoint(state,filename):
    print("====> Saving checkpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint, model):
    print("====> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(dir,bat_size,train_transform,val_transform,num_workers=4,pin_memory=True):
    trainloader = torch.utils.data.DataLoader(
    HumanDataset(dir["train_img"], dir["train_img_mask"],transform=train_transform), 
    batch_size= bat_size, shuffle= True
        , num_workers= num_workers,pin_memory=pin_memory, drop_last=False
    )    
    
    valloader = torch.utils.data.DataLoader(
    HumanDataset(dir["val_img"], dir["val_img_mask"],transform=val_transform), 
    batch_size= bat_size, shuffle= False
        , num_workers= num_workers,pin_memory=pin_memory, drop_last=False
    ) 
    testloader = torch.utils.data.DataLoader(
    HumanDataset(dir["test_img"], dir["test_img_mask"],transform=val_transform), 
    batch_size= bat_size, shuffle= False
        , num_workers= num_workers,pin_memory=pin_memory, drop_last=False
    )        
    return trainloader, valloader, testloader

def check_accuracy(loader,model,device=torch.device("cuda:0")):
    num_correct =0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds==y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)
    acc = num_correct/ num_pixels     
    dice_score =  float(dice_score / len(loader))
    miou = dice_score/(2-dice_score)
    return acc , dice_score,miou
 
import torchvision
import torchvision.transforms as transforms
import torch



def save_predictions_as_imgs(
    loader, model, folder , device = torch.device("cuda:0")
):
    transforms_inv = torch.nn.Sequential(
    transforms.Normalize((0, 0, 0), (1/0.229, 1/0.224, 1/0.225)),
    transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1)),
    )
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        denormalize = transforms_inv(x)
        y = y.unsqueeze(1)
        y = torch.mul(y, denormalize)
        preds = torch.mul(preds  , denormalize)
        torchvision.utils.save_image(
            preds, f"{folder}/{idx}_pred.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")
                

    

