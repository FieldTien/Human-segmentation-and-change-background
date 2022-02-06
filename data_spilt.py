import random
from glob import glob
import re
import pickle

cate = ["Human-Segmentation-Dataset-master","supervise_person","TikTok_tiny","UTFPR-SBD3"]

img_dir = [glob("dataset/%s/images/*"%i, recursive = True) for i in cate]
img_dir = [j for sub in img_dir for j in sub]
print(len(img_dir))
random.shuffle (img_dir)
img_mask_dir = []
for i,file in enumerate(img_dir):
    index = re.split('\\\\|.jpg',file)
    index[0] = index[0].replace("images", "masks/")
    index[2] = '.png' 
    img_mask_dir.append("".join(index))
lergth = len(img_dir)
train_img,val_img,test_img = img_dir[:int(0.8*lergth)],img_dir[int(0.8*lergth):int(0.9*lergth)],img_dir[int(0.9*lergth):]
train_img_mask,val_img_mask,test_img_mask=img_mask_dir[:int(0.8*lergth)],img_mask_dir[int(0.8*lergth):int(0.9*lergth)],img_mask_dir[int(0.9*lergth):]

dir = {'train_img':train_img,'train_img_mask':train_img_mask, 'val_img':val_img, 'val_img_mask':val_img_mask,'test_img':test_img, 'test_img_mask':test_img_mask}      
print("Train legth: ", len(train_img))
print("Valid legth: ", len(val_img))
print("Test legth: ", len(test_img))
with open('tain_val_test_dir.pickle', 'wb') as handle:
    pickle.dump(dir, handle, protocol=pickle.HIGHEST_PROTOCOL)



