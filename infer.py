import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision

from model import GCNet
from misc import loadModel, crf_refine, check_mkdir
import tqdm

class Demo:
    def __init__(self, modelPath="./models/SBU_5000.pt"):
        self.net = GCNet().cuda()
        loadModel(self.net, modelPath)
        self.net.eval()

    def prepare(self, img):
        img = torchvision.transforms.ToTensor()(
            torchvision.transforms.Resize((320, 320))(
                img
            )
        )
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = torchvision.transforms.Normalize(mean, std)(img)
        return img.unsqueeze(0).cuda()

    def postProcess(self, img, pred):
        size = tuple(reversed(img.size))
        pred = torchvision.transforms.ToPILImage()(nn.Sigmoid()(pred["final"][0][0][0]).cpu())
        pred = torchvision.transforms.Resize(size)(pred)
        pred_crf = crf_refine(
            np.array(img),
            np.array(pred)
        )
        return pred_crf

    def detect(self, img_path):
        img = Image.open(img_path)
        pred = self.net(self.prepare(img))
        pred_crf = self.postProcess(img, pred)
        return img, Image.fromarray(pred_crf)

demo = Demo(modelPath="models/SBU_5000.pt")
imgPath = "../dataset/SBU-shadow/SBU-Test/ShadowImages"
outPath = "SBU_output"
imgLst = [x for x in os.listdir(imgPath) if x.endswith(".jpg")]

check_mkdir(outPath)
for x in tqdm.tqdm(imgLst):
    img, pred = demo.detect(os.path.join(imgPath, x))
    pred.save(os.path.join(outPath, x.replace(".jpg", ".png")))

