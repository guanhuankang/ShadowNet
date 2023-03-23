import torch
import torchvision
import progressbar
from torch import optim
import numpy as np
import random

from loss import Loss
from model import ShadowNet
from dataset import getDataLoader
from config import Config
from misc import saveModel
from demo import Demo
# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True

# setup_seed(2021)

def train():
    conf = Config().data()
    print(conf)
    
    dataLoader = getDataLoader()
    net = ShadowNet().cuda()
    net.train()
    # optimizer = optim.SGD([
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #      'lr': 2 * conf['lr']},
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
    #      'lr': conf['lr'], 'weight_decay': conf['weight_decay']}
    # ], momentum=conf['momentum'])
    optimizer = optim.SGD(
        net.parameters(),
        lr=conf["lr"],
        momentum=conf["momentum"],
        weight_decay=conf["weight_decay"]
    )
    loss = Loss().cuda()

    current_iter = 0
    widgets = [progressbar.Percentage(),progressbar.Bar(),progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=conf["max_iter"]).start()
    
    while True:
        if current_iter>=conf["max_iter"]:break
        for index, data in enumerate(dataLoader):
            if current_iter>=conf["max_iter"]:break
            
            # optimizer.param_groups[0]['lr'] = 2 * conf['lr'] * (1 - float(current_iter) / conf['max_iter']
            #                                                     ) ** conf['lr_decay']
            # optimizer.param_groups[1]['lr'] = conf['lr'] * (1 - float(current_iter) / conf['max_iter']
            #                                                 ) ** conf['lr_decay']
            warmup = 500
            if current_iter<=warmup:
                lr = conf["lr"] * float(current_iter) / warmup
            else:
                c = current_iter - warmup
                t = conf["max_iter"] - warmup
                lr = conf["lr"] * (1 - c/t)**conf["lr_decay"]
            optimizer.param_groups[0]["lr"] = lr

            loss(optimizer, net, data, current_iter)
            
            current_iter += 1
            bar.update(current_iter)
            if False and current_iter > 4500 and current_iter%100==0:
                ckp = saveModel(net, aux=str(current_iter))
                demo = Demo(modelPath=ckp)
                demo.eval(imgPath="../dataset/SBU-shadow/SBU-Test/ShadowImages",
                          outPath="temp/iter{}".format(current_iter),
                          gtPath="../dataset/SBU-shadow/SBU-Test/ShadowMasks",
                          name="iter{}".format(current_iter))
    saveModel(net, aux=str(conf["max_iter"]))

if __name__ == '__main__':
    train()

