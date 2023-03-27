import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision

from model import ShadowNet
from misc import loadModel, crf_refine, check_mkdir
import tqdm

from tools.evaluation import Evaluation

class Demo:
    def __init__(self, modelPath="./models/SBU_5000.pt"):
        self.net = ShadowNet().cuda()
        loadModel(self.net, modelPath)
        self.net.eval()

    def prepare(self, img):
        img = torchvision.transforms.ToTensor()(
            torchvision.transforms.Resize((384, 384))(
                img
            )
        )
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = torchvision.transforms.Normalize(mean, std)(img)
        return img.unsqueeze(0).cuda()

    def postProcess(self, img, pred):
        size = tuple(reversed(img.size))
        pred = torchvision.transforms.ToPILImage()(nn.Sigmoid()(pred["pred"][0][0]).cpu())
        pred = torchvision.transforms.Resize(size)(pred)
        pred_crf = crf_refine(
            np.array(img),
            np.array(pred)
        )
        return pred_crf

    def detect(self, img_path):
        img = Image.open(img_path).convert("RGB")
        pred = self.net(self.prepare(img))
        pred_crf = self.postProcess(img, pred)
        return img, Image.fromarray(pred_crf)

    def eval(self, imgPath, outPath, gtPath, name="evaluation"):
        imgLst = [x for x in os.listdir(imgPath) if x.endswith(".jpg") or x.endswith(".png")]
        check_mkdir(outPath)
        for x in tqdm.tqdm(imgLst):
            img, pred = self.detect(os.path.join(imgPath, x))
            pred.save(os.path.join(outPath, x.replace(".jpg", ".png")))
        e = Evaluation(gtPath, outPath, name)
        e.calc()
        r = e.echo()
        print(r, flush=True)
        return r