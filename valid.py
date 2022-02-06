import pickle 

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import time
from thop.profile import profile
from thop import clever_format
 

def valid_main(cfg):
    IMAGE_HEIGHT = cfg.IMAGESIZE[0]  
    IMAGE_WIDTH = cfg.IMAGESIZE[1] 
    NUM_WORKERS = cfg.NUM_WORKERS
    PIN_MEMORY = cfg.PIN_MEMORY
    DEVICE = cfg.DEVICE
    BATCH_SIZE = cfg.BATCH_SIZE
    PATH = cfg.Model_PATH
    if cfg.MODEL == "MobileV2":
        from  models_define.UNet_mobileV2 import load_MobileV2_UNET
        model = load_MobileV2_UNET()
    elif cfg.MODEL == "VGG16":   
        from  models_define.UNet_VGG16 import load_VGG16_UNET
        model = load_VGG16_UNET() 
    else:
        print("No model selected")
    model = model.to(DEVICE)
    #model = UNET_VGG16(in_channels=3, out_channels=1).to(DEVICE)

    with open('tain_val_test_dir.pickle', 'rb') as handle:
        dir = pickle.load(handle)

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
    _ , val_loader, test_loader = get_loaders(
        dir,
        BATCH_SIZE,
        val_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    

    
    
    load_checkpoint(torch.load(PATH), model)
    
    
    start = time.time()
    val_acc,val_dicescore,val_miou = check_accuracy(val_loader, model, device=DEVICE)
    print(f"Accuracy {val_acc*100:.2f}%")
    print(f"Dice score {val_dicescore*100:.2f}%")
    print(f"MIOU {val_miou*100:.2f}%")
    test_acc,test_dicescore,test_miou = check_accuracy(test_loader, model, device=DEVICE)
    print(f"Accuracy {test_acc*100:.2f}%")
    print(f"Dice score {test_dicescore*100:.2f}%")  
    print(f"MIOU {test_miou*100:.2f}%")
    print('\n\nValid and Test inferece time: %.4f\n'%(time.time() - start)) 
    
    save_predictions_as_imgs(
        val_loader, model, folder="bestmodel_inference/", device=DEVICE
        )
    
    dsize = (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    inputs = torch.randn(dsize).to(DEVICE)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    print("MACS:",macs,"\nparams:",params)
    

if __name__ == "__main__":
    valid_main()    
    
    
    
        
        
    