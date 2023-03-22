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

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True

# setup_seed(2021)

def train():
    conf = Config().data()
    
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
            lr = conf["lr"] * (1 - float(current_iter)/(conf["max_iter"]+10))**conf["lr_decay"]
            optimizer.param_groups[0]["lr"] = lr

            loss(optimizer, net, data, current_iter)
            
            current_iter += 1
            bar.update(current_iter)
            if current_iter > 4500 and current_iter%100==0:
                saveModel(net, aux=str(current_iter))
    saveModel(net, aux=str(conf["max_iter"]))

if __name__ == '__main__':
    train()

