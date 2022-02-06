import argparse
from pickle import TRUE
import wget
import os
import torch
import gdown
import zipfile

def config():
    
    parser = argparse.ArgumentParser(description="Settings of Human Segmentation")
    parser.add_argument('--mode', type=str, choices=["train", "evaluation","inference","Real Time inference","download"],
                        default="evaluation", help="train  or evaluation")
    parser.add_argument('--DEVICE', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device for training and testing")
    parser.add_argument('--IMAGESIZE', type=tuple, default=(256,256), help="model image size")
    parser.add_argument('--MODEL', type=str, choices=["MobileV2", "VGG16"],
                        default="MobileV2", help="select the backbone of UNET")
    #parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")
    parser.add_argument('--BATCH_SIZE', type=int, default=24, help="batch size for training")
    parser.add_argument('--LR', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--EPOCH', type=int, default=150, help="epochs for training")    
    parser.add_argument('--NUM_WORKERS', type=int, default=4, help="number of workers in dataloader")   
    parser.add_argument('--PIN_MEMORY', type=bool, default=True, help="pin memory in dataloader")     
    parser.add_argument('--autoadjust_model_inputsize', type=bool, default=False, help="Auto-adjust the model input size when inferece") 
    parser.add_argument('--background', type=str, default="test_img/test_backgroud.jpg", help="background PATH") 
    parser.add_argument('--CAMERA_SIZE', type=tuple, default=(480,640), help="Camera size when inferece") 
    parser.add_argument('--change_background_img', type=str, default="test_img/test_2.jpg", help="Image if using change_background") 
    parser.add_argument('--change_background_output', type=str, default="test_img/test_2_result.jpg", help="Output dir if using change_background") 
    parser.add_argument('--Model_PATH', type=str, default="models/UNet_mobileV2.pth", help="Model Path") 
    parser.add_argument('--Out_threshold', type=float, default=0.5, help="Output probability threshold") 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = config()
    cfg.mode = "evaluation"
    #cfg.MODEL =  "VGG16"
    cfg.Model_PATH = "models/UNet_MobileV2_256x256.pth"
    cfg.autoadjust_model_inputsize = False
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.IMAGESIZE = (256,256)
    cfg.CAMERA_SIZE = (480,640)
    
    if(cfg.mode == "train"):
        from train import train_main
        train_main(cfg)
    elif(cfg.mode == "evaluation"):
        from valid import valid_main
        valid_main(cfg)
    elif(cfg.mode == "inference"):
        from inference import change_background
        change_background(cfg)
    elif(cfg.mode == "Real Time inference"):
        from inference import Real_time_inference
        Real_time_inference(cfg)
    elif(cfg.mode == "download"):
        file_path = "models.zip"
        if os.path.isfile(file_path):
            print("Model Zip file has already exists")
        else:    
            url = 'https://drive.google.com/uc?id=1S11udQQSJ3faIGjLxUkgbrDVo6D4ayeU'
            output = file_path
            gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("")    
            
        file_path = "dataset.zip"
        if os.path.isfile(file_path):
            print("Data Zip file has already exists")
        else:    
            url = 'https://drive.google.com/uc?id=1btPq1ICmYs2fCwA49kS15VLt8KpQi5j_'
            output = file_path
            gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("")        
            
    else:
        print("Mode is wrong")    
 