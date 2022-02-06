
import torch.utils.data as data
import numpy as np
from PIL import Image


class HumanDataset(data.Dataset):
    def __init__(self, img_dir, mask_dir,transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
    def __getitem__(self, index):
        img_dir = self.img_dir[index]
        mask_dir = self.mask_dir[index]
        img_dir = np.array(Image.open(img_dir).convert('RGB'))
        mask_dir = Image.open(mask_dir).convert('L')
        mask_dir = np.array(mask_dir,dtype=np.float32)  
        mask_dir = np.array(mask_dir,dtype=np.float32)  
        mask_dir[mask_dir > 0 ] = 1.0
    
        if self.transform is not None:      
            augmentations = self.transform(image=img_dir, mask=mask_dir)
            img_dir = augmentations["image"]
            mask_dir = augmentations["mask"]
     
        return img_dir,mask_dir
    def __len__(self):
        return len(self.img_dir)    
    

