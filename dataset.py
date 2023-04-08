from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import numpy as np
import imageio
import os

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir,mask_dir,mask_file_tag="",
                 image_only_transforms=None,image_mask_transforms=[]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = [file for file in os.listdir(image_dir) 
                            if os.path.splitext(file)[1] == ".JPG"]
        self.image_only_transforms = image_only_transforms
        self.image_mask_transforms = image_mask_transforms
        self.mask_file_tag = mask_file_tag
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, 
                            f"{self.image_names[idx][:-4]}{self.mask_file_tag}.gif")
        image = read_image(image_path).type(torch.FloatTensor)
        mask = np.array(imageio.mimread(mask_path))

        mask = torch.as_tensor(mask).float().contiguous()
                
        if self.image_only_transforms:
            image = self.image_only_transforms(image)
        for t in self.image_mask_transforms:
            image,mask = t(image,mask)
        return image, mask    