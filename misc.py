import os
from datetime import datetime
import time
import torchvision
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
import config

def log(*args):
    with open("log.txt", "a+") as f:
        msg = [str(_) for _ in args]
        f.write(str(msg)+"\n")

def check_mkdir(path):
    os.makedirs(path, exist_ok=True)
        
def saveModel(model, aux=""):
    aux = str(aux)
    D = "models"
    check_mkdir(D)
    conf = config.Config()
    path = os.path.join(D, "%s_%s.pt"%(conf.dataset_name, aux))
    torch.save(model.state_dict(), path)
    return path

def metric(mask, gt):
    mask = (mask+0.5).int()
    gt = (gt+0.5).int()
    TP = (mask*gt==1).sum()*1.0
    TN = (mask+gt==0).sum()*1.0
    FP = ((mask-1)==gt).sum()*1.0
    FN = (mask==(gt-1)).sum()*1.0

    return "BER:%f Accu:%f Shadow:%f Nonshadow:%f"%(
        (1. - 0.5*(TP/(TP+FN)+TN/(TN+FP)))*100,
        (TP+TN)/(TP+TN+FP+FN),
        100*(1. - TP/(TP+FN)),
        100*(1. - TN/(TN+FP))
    )
    
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def loadModel(model, path):
    model.load_state_dict(torch.load(path))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')
