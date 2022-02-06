from locale import normalize
import pickle
from turtle import shape 
import torch.nn.functional as F
import torch
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from utils import (
    load_checkpoint,

)
import cv2
from PIL import Image
import numpy as np
def change_background(cfg):
    valid_img = cfg.change_background_img
    background = cfg.background
    autoadjust_model_inputsize = cfg.autoadjust_model_inputsize
    
    DEVICE = cfg.DEVICE
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
    print(PATH)
    load_checkpoint(torch.load(PATH), model)
    model.eval()
    valid_img = Image.open(valid_img).convert('RGB')
    target_size = valid_img.size
    if autoadjust_model_inputsize:
        IMAGE_WIDTH = target_size[0] -  (target_size[0]%(2**5))
        IMAGE_HEIGHT = target_size[1] -  (target_size[1]%(2**5))
    else:
        IMAGE_HEIGHT = cfg.IMAGESIZE[0]  
        IMAGE_WIDTH = cfg.IMAGESIZE[1]
    print("IMAGE_WIDTH: ",IMAGE_WIDTH)
    print("IMAGE_HEIGHT: ",IMAGE_HEIGHT)
    normalize = torch.nn.Sequential(
        transforms.Resize([IMAGE_HEIGHT,IMAGE_WIDTH]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    valid_model = transforms.functional.to_tensor(valid_img).unsqueeze(0)
    valid_model = normalize(valid_model).to(DEVICE)

    with torch.no_grad():
        preds = torch.sigmoid(model(valid_model))
    preds = F.interpolate(preds, size=(target_size[1],target_size[0]),mode='bilinear',align_corners=True)
    preds = (preds > cfg.Out_threshold).float().squeeze(0) 
    
     
    
    valid_img = transforms.functional.to_tensor(valid_img).to(DEVICE)
    background = Image.open(background).convert('RGB').resize(target_size)
    background = transforms.functional.to_tensor(background).to(DEVICE)
    preds_inverse = torch.abs((preds-1))
    background_ectracted = torch.mul(preds_inverse, background) 
    ectracted = torch.mul(preds, valid_img)
    transform = background_ectracted+ectracted
    transform = transforms.functional.to_pil_image(transform)
    transform.save(cfg.change_background_output)

    return transform


def Real_time_inference(cfg):
    CAMERA_SIZE = cfg.CAMERA_SIZE
    background = cfg.background
    autoadjust_model_inputsize = cfg.autoadjust_model_inputsize
    PATH = cfg.Model_PATH
    DEVICE = cfg.DEVICE
    if cfg.MODEL == "MobileV2":
        from  models_define.UNet_mobileV2 import load_MobileV2_UNET
        model = load_MobileV2_UNET()
    elif cfg.MODEL == "VGG16":   
        from  models_define.UNet_VGG16 import load_VGG16_UNET
        model = load_VGG16_UNET() 
    else:
        print("No model selected")
    model = model.to(DEVICE)
  
    load_checkpoint(torch.load(PATH), model)
    background = Image.open(background).convert('RGB').resize((CAMERA_SIZE[1],CAMERA_SIZE[0]))
    background = transforms.functional.to_tensor(background).to(DEVICE)
    print(background.shape)
    if autoadjust_model_inputsize:
        IMAGE_WIDTH = CAMERA_SIZE[1] -  (CAMERA_SIZE[1]%(2**5))
        IMAGE_HEIGHT = CAMERA_SIZE[0] -  (CAMERA_SIZE[0]%(2**5))
    else:
        IMAGE_HEIGHT = cfg.IMAGESIZE[0]  
        IMAGE_WIDTH = cfg.IMAGESIZE[1]
    normalize = torch.nn.Sequential(
        transforms.Resize([IMAGE_HEIGHT,IMAGE_WIDTH]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[1])
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    cv2.namedWindow("Real time segmentation", cv2.WINDOW_AUTOSIZE); 
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        valid_img = transforms.functional.to_tensor(frame).unsqueeze(0)
        with torch.no_grad():   
           preds = torch.sigmoid(model(normalize(valid_img).to(DEVICE)))
        preds = F.interpolate(preds, size=(CAMERA_SIZE[0],CAMERA_SIZE[1]),mode='bilinear',align_corners=True)
        preds = (preds > cfg.Out_threshold).float().squeeze(0) 
        preds_inverse = torch.abs((preds-1))
        background_ectracted = torch.mul(preds_inverse, background) 
        ectracted = torch.mul(preds, valid_img.to(DEVICE))
        transform = (background_ectracted+ectracted).squeeze(0).permute(1,2,0).cpu().detach().numpy()
        transform = cv2.cvtColor(transform, cv2.COLOR_RGB2BGR)
        cv2.imshow('Real time segmentation', transform)
        if cv2.waitKey(1) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

