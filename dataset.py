import os
import PIL.Image as Image
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import random

from config import Config


class ImageFolder(data.Dataset):
    def __init__(self):
        c = Config()
        conf = c.data()
        self.img_path = conf["data_paths"][0]
        self.gt_path = conf["data_paths"][1]
        if c.dataset_name=="SBU":
            name_list = []
            with open("scores.txt", "r") as f:
                for line in f.readlines():
                    name_list.append(line.split()[0])
            self.name_list = name_list[0:2500]
        else:
            self.name_list = [x for x in os.listdir(self.img_path) if x.endswith(".jpg") or x.endswith(".png")]
        self.scale = conf["scale"]
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        cvt = lambda x: torchvision.transforms.ToTensor()(
            torchvision.transforms.Resize(self.scale)(x)
        )
        img = cvt(Image.open(os.path.join(self.img_path, name)))
        gt = cvt( Image.open(os.path.join(self.gt_path, name.replace(".jpg", ".png"))) )

        ## Random Flip
        if random.random()<=0.5:
            img = torchvision.transforms.functional.hflip(img)
            gt = torchvision.transforms.functional.hflip(gt)

        ## randomly corrupt images
        n = 16
        h, w = self.scale[0]//n, self.scale[1]//n
        corruption = torch.rand(1, 1, h, w).lt(1.0).float()
        corruption_mask = torch.nn.functional.interpolate(corruption, size=self.scale, mode="nearest")[0]
        img = img * corruption_mask

        return img, gt

def getDataLoader():
    bs = Config().data()["batch_size"]
    imageFolder = ImageFolder()
    return DataLoader(imageFolder, batch_size=bs, num_workers=8, shuffle=True)

if __name__=="__main__":
    imageFolder = ImageFolder()
    for i in range(10):
        imageFolder[i]