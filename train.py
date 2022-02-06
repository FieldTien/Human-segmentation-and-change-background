import pickle 

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#from models_define import UNET , UNET_VGG16




def train_fn(loader, model, optimizer, loss_fn, scaler,DEVICE):
    model.train()
    loop = tqdm(loader)
    for data,targets in loop:
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)
        
        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()    
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
def train_main(cfg):
    
    IMAGE_HEIGHT = cfg.IMAGESIZE[0]  
    IMAGE_WIDTH = cfg.IMAGESIZE[1] 
    NUM_WORKERS = cfg.NUM_WORKERS
    PIN_MEMORY = cfg.PIN_MEMORY
    DEVICE = cfg.DEVICE
    BATCH_SIZE = cfg.BATCH_SIZE
    LEARNING_RATE = cfg.LR
    NUM_EPOCHS = cfg.EPOCH
    PATH =  cfg.Model_PATH
    if cfg.MODEL == "MobileV2":
        from  models_define.UNet_mobileV2 import load_MobileV2_UNET
        model = load_MobileV2_UNET()
    elif cfg.MODEL == "VGG16":   
        from  models_define.UNet_VGG16 import load_VGG16_UNET
        model = load_VGG16_UNET() 
    else:
        print("No model selected")

    
    model = model.to(DEVICE)
    with open('tain_val_test_dir.pickle', 'rb') as handle:
        dir = pickle.load(handle)

    train_transform = A.Compose(
            [
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.1),
                A.RandomResizedCrop(width=IMAGE_HEIGHT,height=IMAGE_WIDTH),
                A.Normalize(
                    mean=[[0.485, 0.456, 0.406]],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[[0.485, 0.456, 0.406]],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_loader, val_loader, test_loader = get_loaders(
        dir,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    #model = UNET_VGG16(in_channels=3, out_channels=1).to(DEVICE)
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()
    print('Number of Weight %d' %(params))   
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    scaler = torch.cuda.amp.GradScaler()  
    best_dicescore = 0  
    for epoch in range(NUM_EPOCHS):
        print(f"EPOCH: ",epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler,DEVICE)
        val_acc,val_dicescore,val_miou = check_accuracy(val_loader, model, device=DEVICE)
        print(f"Accuracy {val_acc*100:.2f}%")
        print(f"Dice score {val_dicescore*100:.2f}%")
        print(f"MIOU {val_miou*100:.2f}%")
        scheduler.step()
        if (best_dicescore < val_dicescore):
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint,PATH)
            best_dicescore = val_dicescore
            
            save_predictions_as_imgs(
            val_loader, model, folder="bestmodel_inference/", device=DEVICE
            )
    load_checkpoint(torch.load(PATH), model)
    test_acc,test_dicescore,test_miou = check_accuracy(test_loader, model, device=DEVICE)
    print(f"Accuracy {test_acc*100:.2f}%")
    print(f"Dice score {test_dicescore*100:.2f}%")  
    print(f"MIOU {test_miou*100:.2f}%")
        
        
    
    

if __name__ == "__main__":
    train_main()