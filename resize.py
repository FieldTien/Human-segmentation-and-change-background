from PIL import Image
import numpy as np
from glob import glob
HEIGHT = 256
WIDTH = 256

img_dir = glob("dataset/supervise_person/images/*", recursive = True)
for img in img_dir:
    img_pil = Image.open(img).convert('RGB').resize((HEIGHT, WIDTH))
    img_pil.save(img)
img_dir = glob("dataset/supervise_person/masks/*", recursive = True)
for img in img_dir:
    img_pil = Image.open(img).convert('L').resize((HEIGHT, WIDTH))
    img_pil.save(img)
   
img_dir = glob("dataset/Human-Segmentation-Dataset-master/images/*", recursive = True)
for img in img_dir:
    img_pil = Image.open(img).convert('RGB').resize((HEIGHT, WIDTH))
    img_pil.save(img)
    
img_dir = glob("dataset/Human-Segmentation-Dataset-master/masks/*", recursive = True)
for img in img_dir:
    img_pil = Image.open(img).convert('L').resize((HEIGHT, WIDTH))
    img_pil.save(img)    

