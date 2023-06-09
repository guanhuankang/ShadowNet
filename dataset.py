import os
import PIL.Image as Image
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import random
import numpy as np
from config import Config


class ImageFolder(data.Dataset):
    def __init__(self):
        c = Config()
        conf = c.data()
        self.img_path = conf["data_paths"][0]
        self.gt_path = conf["data_paths"][1]
        if c.dataset_name=="SBU":
            name_list = []
            with open("sbu_config/scores.txt", "r") as f:
                for line in f.readlines():
                    name_list.append(line.split()[0])
            self.name_list = name_list[0:3200] + name_list[0:2500] + name_list[0:1000]
        else:
            self.name_list = [x for x in os.listdir(self.img_path) if x.endswith(".jpg") or x.endswith(".png")]
        print("dataset len", len(self.name_list))
        self.scale = conf["scale"]
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        cvt = lambda x: torchvision.transforms.ToTensor()(
            torchvision.transforms.Resize(self.scale)(x)
        )
        img = cvt(Image.open(os.path.join(self.img_path, name)).convert("RGB"))
        gt = cvt( Image.open(os.path.join(self.gt_path, name.replace(".jpg", ".png"))).convert("L") )

        ## Random Flip
        if random.random()<=0.5:
            img = torchvision.transforms.functional.hflip(img)
            gt = torchvision.transforms.functional.hflip(gt)

        ## RandomCrop
        # a = np.random.rand() * 0.5 + 1.0
        # img = F.interpolate(img.unsqueeze(0), scale_factor=a, mode="bilinear")
        # gt = F.interpolate(gt.unsqueeze(0), scale_factor=a, mode="nearest")
        # catcon = torchvision.transforms.RandomCrop(self.scale)(torch.cat([img, gt], dim=1))[0]
        # img, gt = catcon[0:3], catcon[3::]

        ## Random Brightness
        if random.random()<=0.5:
            img = torchvision.transforms.ColorJitter(brightness=0.5)(img)

        ## randomly corrupt images
        n = 16
        h, w = self.scale[0]//n, self.scale[1]//n
        corruption = torch.rand(1, 1, h, w).lt(0.85).float()
        corruption_mask = torch.nn.functional.interpolate(corruption, size=self.scale, mode="nearest")[0]
        img = img * corruption_mask

        return img, gt

def getDataLoader():
    bs = Config().data()["batch_size"]
    imageFolder = ImageFolder()
    return DataLoader(imageFolder, batch_size=bs, num_workers=8, shuffle=True)

if __name__=="__main__":
    imageFolder = ImageFolder()
    for i in range(2000, 2010):
        imageFolder[i]