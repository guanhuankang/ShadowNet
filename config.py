import sys, os

class Config:
    def __init__(self):
        ## Configure the following three variables manually ##
        self.train_img_path = "../dataset/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages"
        self.train_gt_path = "../dataset/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks"
        self.dataset_name = "SBU" ## OR ISTD
        ####################################################
        self.init()

    def init(self):
        if self.dataset_name=="SBU":
            self.shadow_recall_factor = 1.10
        else:
            self.shadow_recall_factor = 1.00

    def data(self):
        return {
            "data_paths":[ self.train_img_path, self.train_gt_path],##[img, gt]
            "batch_size": 16,
            "lr": 5e-3,
            "momentum":0.9,
            "weight_decay":5e-4,
            "max_iter": 2500,
            "lr_decay":0.9,
            "scale":(416,416)
        }
    
    def err(self, index, aux=None):
        return {
            0:"#0:Pls Check config.data_paths! The amount of files in each folder aren't matched.",
            1:"#1:The data_paths's length should be larger than 0.",
            2:"#2:Pls Check the pictures in the given folder[data_paths]. Names are not matched among these folders"
        }[index]+" aux:"+str(aux)
