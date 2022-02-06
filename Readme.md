# Human segmentation and change background

## Dataset
* [Supervisely Person](https://hackernoon.com/releasing-supervisely-person-dataset-for-teaching-machines-to-segment-humans-1f1fc1f28469): Extracted 5060 images
* [Human Segmentation Dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset): Extracted 299 images
* [TikTok Dataset ](https://paperswithcode.com/dataset/tiktok-dataset): Extracted 7500 images
* [UTFPR-SBD3](https://github.com/bioinfolabic/UTFPR-SBD3): Extracted 3045 images

In Dataset, I deleted the data like following figures

<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/dataset0.png" width="300" height="300">

<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/dataset1.png" width="300" height="300">

**You can download the full pre-processed data in [here](https://drive.google.com/file/d/1btPq1ICmYs2fCwA49kS15VLt8KpQi5j_/view?usp=sharing)**

# Model Result
* ImageNet finetuning VGG16,MobileV2 as encoder, and freezed the encoder parameters.
* Still trying the other Encoders like mobileV3,effienctNet,RegNet and the UNet3+ archihteture.

### Training image size 256x256(200 Epochs)
|          | Valid  Acc  | Valid MIoU | Test  Acc |Test MIoU | MACS |Parameters |
| -------- | --------   | --------| -------- | --------       | --------        |--------  | 
| UNET_VGG16_256x256| 98.47%     | 94.99% |98.40%   |94.76%            |44.656G          |29.273M |
| UNET_MobileV2_256x256 | 98.19%     | 94.05%  |98.16%   |94.00%            |2.166G            |5.108M |

### Training image size 512x512(200 Epochs)
|          | Valid  Acc  | Valid MIoU | Test  Acc |Test MIoU | MACS |Parameters |
| -------- | --------   | --------| -------- | --------       | --------        |--------  | 
| UNET_MobileV2_512x512| 98.35%     | 94.57% |98.32%  |94.44%            |11.316G          |5.108M |

# How to run 
## requirements
pytorch albumentations thop numpy PIL wget
## Download Dataset and Pretrained Models
Here provide the preprocessed data and pretrained Models
```
python main.py --mode download                   
```

## Train

```
python main.py --mode          train      
               --data_dir_head [Datapath] 
               --BATCH_SIZE    [BATCH_SIZE] 
               --LR            [Learning Rate] 
               --EPOCH         [Epochs] 
               --backbone      [Feature map of Conv in VGG19]
               --latent_dim    [Latent size of CAE] 
               --classes       [Default is all] 
```

## Evaluate the Valid and Test Set
```
python main.py --mode        evaluation    
               --IMAGESIZE   [Model input size Default is (256,256)] 
               --Model       [Default is "MobileV2"] 
               --Model_PATH  [Default is "models/UNet_mobileV2.pth] 
               --BATCH_SIZE  [Default is 24] 
               
```
The folder bestmodel_inference/ will save the validation correct mask and prediction mask which combined with input image and black background 
**Correct Mask with input images**
<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/dataset0.png" width="300" height="300">
**Prediction Mask with input images**
<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/dataset1.png" width="300" height="300">

## Inference the model
```
python main.py --mode                     inference    
               --background              [background path] 
               --change_background_img   [Input image path] 
               --change_background_output[Output image path]
               --Model                   [Default is "MobileV2"]
               --Model_PATH              [Default is "models/UNet_mobileV2.pth] 
               --autoadjust_model_inputsize [Default is False] 
               --IMAGESIZE               [Model input size Default is (256,256)]   
```
* If autoadjust_model_inputsize = True, it will adjust the the IMAGESIZE automatically. when input size is (1695,960) then model input will resize to (1664,960) 
* If you want to set autoadjust_model_inputsize = True, I reccomand to use the model UNet_mobileV2_512X512.pth which is trained with 512x512 inputs images

**Example1 Use the mobileV2 Unet which trained by 256x256 pixel images**

```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "inference"
    cfg.Model_PATH = "models/UNet_mobileV2_256x256.pth"
    cfg.IMAGESIZE = (256,256)
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.change_background_img = "test_img/test_0.jpg"
    cfg.change_background_output = "test_img/test_0_result.jpg"
```

Input Image              |  Output image
:-------------------------:|:-------------------------:
![](https://i.imgur.com/CFcm1WB.jpg)  |  ![](https://i.imgur.com/czm949m.jpg)

**Example2 Use the mobileV2 Unet which trained by 512x512 pixel images**

```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "inference"
    cfg.Model_PATH = "models/UNet_mobileV2_512x512pth"
    cfg.IMAGESIZE = (512,512)
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.change_background_img = "test_img/test_0.jpg"
    cfg.change_background_output = "test_img/test_0_result.jpg"
```

Input Image              |  Output image
:-------------------------:|:-------------------------:
![](https://i.imgur.com/CFcm1WB.jpg )  |  ![](https://i.imgur.com/NxX2Udu.jpg )

**Example3 Use the mobileV2 Unet which trained by 512x512 pixel images and inferece autoadjust_model_inputsize = True** 

```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "inference"
    cfg.Model_PATH = "models/UNet_mobileV2_512x512pth"
    cfg.autoadjust_model_inputsize = True
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.change_background_img = "test_img/test_0.jpg"
    cfg.change_background_output = "test_img/test_0_result.jpg"
```

Input Image              |  Output image
:-------------------------:|:-------------------------:
![](https://i.imgur.com/CFcm1WB.jpg )  |  ![](https://i.imgur.com/Uv6miUp.jpg )

## Real time Inference
### Using UNet_VGG16_256x256 with 256x256 size model input
```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "Real Time inference"
    cfg.MODEL =  "VGG16"
    cfg.Model_PATH = "models/UNet_VGG16_256x256.pth"
    cfg.IMAGESIZE = (256,256)
    cfg.autoadjust_model_inputsize = False
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.CAMERA_SIZE = (480,640)
```
<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/vgg16.gif" width="300" height="300">

### Using UNet_MobileV2_256x256 with 256x256 size model input
```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "Real Time inference"
    cfg.MODEL =  "MobileV2"
    cfg.Model_PATH = "models/UNet_MobileV2_256x256.pth"
    cfg.IMAGESIZE = (256,256)
    cfg.autoadjust_model_inputsize = False
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.CAMERA_SIZE = (480,640)
```

<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/mobile256.gif" width="300" height="300">
 
### Using UNet_MobileV2_512x512 with autoadjust_model_inputsize =True
```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "Real Time inference"
    cfg.MODEL =  "MobileV2"
    cfg.Model_PATH = "models/UNet_MobileV2_512x512.pth"
    cfg.autoadjust_model_inputsize = True
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.CAMERA_SIZE = (480,640)
```
<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/Mobile512.gif" width="300" height="300">

### Using UNet_MobileV2_512x512 with changing the output threshold

```
if __name__ == "__main__":
    cfg = config()
    cfg.mode = "Real Time inference"
    cfg.MODEL =  "MobileV2"
    cfg.Model_PATH = "models/UNet_MobileV2_512x512.pth"
    cfg.autoadjust_model_inputsize = True
    cfg.background = "test_img/test_backgroud.jpg"
    cfg.CAMERA_SIZE = (480,640)
    cfg.Out_threshold = 0.9
```
<img src="https://github.com/FieldTien/Human-segmentation-and-change-background/blob/main/readme_pic/threshold.gif" width="300" height="300">

# Reference
https://github.com/aladdinpersson/Machine-Learning-Collection

https://github.com/thuyngch/Human-Segmentation-PyTorch

https://www.researchgate.net/publication/341749157_Mobile-Unet_An_efficient_convolutional_neural_network_for_fabric_defect_detection
