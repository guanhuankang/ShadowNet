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
        conf = Config().data()

        self.data_list = []
        data_exts = [".jpg", ".png", ".JPG", ".PNG"]
        for p in conf["data_paths"]:
            LL = os.listdir(p)
            L = []
            for _ in LL:
                if _[-4::] in data_exts:L.append(_)
            L.sort()
            self.data_list.append(L)

        ## check
        cnt = len(self.data_list)
        if cnt<=0:
            print(conf.err(1));exit(0)
        tot = len(self.data_list[0])
        for l in range(1,cnt):
            if len(self.data_list[l])!=tot: print(conf.err(0));exit(0)
            for _ in range(tot):
                if self.data_list[0][_][:-4]!=self.data_list[l][_][:-4]:print(conf.err(1));exit(0)
        
        self.data_paths = conf["data_paths"]
        self.scale = conf["scale"]
    
    def __len__(self):
        return len(self.data_list[0])
    
    def __getitem__(self, index):
        cnt = len(self.data_list)
        tot = len(self.data_list[0])

        ret = []
        for i in range(cnt):
            p = os.path.join( self.data_paths[i], self.data_list[i][index] )
            img = Image.open(p)
            ret.append(
                torchvision.transforms.ToTensor()(
                    torchvision.transforms.Resize(self.scale)(
                        img
                    )
                )
            ) ## 0-1
        
        ## Random Flip
        if random.random()<=0.5:
            for _ in range(len(ret)):
                ret[_] = torchvision.transforms.functional.hflip(ret[_])

        ## randomly corrupt images
        n = 16
        H, W = self.scale
        h, w = H//n, W//n
        corruption = torch.rand(1, 1, h, w).lt(0.85).float()
        corruption_mask = torch.nn.functional.interpolate(corruption, size=(H, W), mode="nearest")[0]
        # corruption_mask = (corruption_mask + ret[1]).gt(0.5).float() ## filter shadow area
        ret[0] = ret[0] * corruption_mask

        return ret

def getDataLoader():
    bs = Config().data()["batch_size"]
    imageFolder = ImageFolder()
    return DataLoader(imageFolder, batch_size=bs, num_workers=8, shuffle=True)

if __name__=="__main__":
    imageFolder = ImageFolder()
    for i in range(10):
        imageFolder[i]