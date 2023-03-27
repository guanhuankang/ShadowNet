import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

from config import Config

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = self.block2(block1)
        return block2

class head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )
    def forward(self, x):
        return self.conv_1x1(x)

class LocalShadowDetector(nn.Module):
    def __init__(self, size):
        super(LocalShadowDetector, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d( 64, 32, kernel_size=(1,1), bias=False, groups=32 ), 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d( 32, 32, kernel_size=(7,7), padding=(3, 3), bias=False, groups=8 ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False, groups=32)
        )
        self.pred = nn.Conv2d(32, 1, 1, bias=False)
        self.size = size
    
    def forward(self, x):
        return self.pred( F.interpolate(self.block(x), size=self.size, mode="bilinear") + \
            F.interpolate(self.conv(x), size=self.size, mode="bilinear"))

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnext101_32x8d(pretrained=True)
        # backbone = models.resnet50(pretrained=True)
        print("adopt pretrained R101 weights")
        # backbone = models.resnext101_32x8d(weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        # print("adopt ResNeXt101_32X8D_Weights.IMAGENET1K_V1")


        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.layer1 = nn.Sequential(
            backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.reduction4 = nn.Sequential(
            nn.Conv2d( 2048, 512, 3, padding=1, bias=False ), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d( 512, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction3 = nn.Sequential(
            nn.Conv2d( 1024, 512, 3, padding=1, bias=False ), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d( 512, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction2 = nn.Sequential(
            nn.Conv2d( 512, 256, 3, padding=1, bias=False ), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d( 256, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction1 = nn.Sequential(
            nn.Conv2d( 256, 128, 3, padding=1, bias=False ), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d( 128, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction0 = nn.Sequential(
            nn.Conv2d( 64, 64, 3, padding=1, bias=False ), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d( 64, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.fusion3 = ConvBlock(64, 32)
        self.fusion2 = ConvBlock(64, 32)
        self.fusion1 = ConvBlock(96, 32)
        self.fusion0 = ConvBlock(128, 32)

        self.pred3 = head(32)
        self.pred2 = head(32)
        self.pred1 = head(32)
        self.pred0 = head(32)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        y0 = self.reduction0(layer0)
        y1 = self.reduction1(layer1)
        y2 = self.reduction2(layer2)
        y3 = self.reduction3(layer3)
        y4 = self.reduction4(layer4)

        r3 = self.fusion3( torch.cat( (
            F.interpolate(y4, size=y3.shape[2:4], mode="bilinear"),
            y3
        ), dim = 1 ) )
        r2 = self.fusion2( torch.cat( (
            F.interpolate(r3, size=y2.shape[2:4], mode="bilinear"),
            y2
        ), dim = 1 ) )
        r1 = self.fusion1( torch.cat( (
            F.interpolate(r3, size=y1.shape[2:4], mode="bilinear"),
            F.interpolate(r2, size=y1.shape[2:4], mode="bilinear"),
            y1
        ), dim = 1 ) )
        r0 = self.fusion0( torch.cat( (
            F.interpolate(r3, size=y0.shape[2:4], mode="bilinear"),
            F.interpolate(r2, size=y0.shape[2:4], mode="bilinear"),
            F.interpolate(r1, size=y0.shape[2:4], mode="bilinear"),
            y0
        ), dim = 1 ) )

        ss3 = self.pred3( F.interpolate(r3, size=x.shape[2:4], mode="bilinear") )
        ss2 = self.pred2( F.interpolate(r2, size=x.shape[2:4], mode="bilinear") )
        ss1 = self.pred1( F.interpolate(r1, size=x.shape[2:4], mode="bilinear") )
        ss0 = self.pred0( F.interpolate(r0, size=x.shape[2:4], mode="bilinear") )

        cue0 = torch.cat((y0,r0), dim=1)
        cue1 = torch.cat((y1,r1), dim=1)
        cue2 = torch.cat((y2,r2), dim=1)
        cue3 = torch.cat((y3,r3), dim=1)

        return {
            "shadow_scores": [ss0, ss1, ss2, ss3],
            "cues": [cue0, cue1, cue2, cue3]
        }

class DASA(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.LSDForIntersectionBranch = LocalShadowDetector(size=size)
        self.LSDForDivergenceBranch = LocalShadowDetector(size=size)
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, dark_region, shadow_scores, cue):
        hard_dark_region = torch.sign( torch.relu(dark_region) )
        hard_shadow_scores = torch.sign( torch.relu(shadow_scores) )
        Od = torch.relu(hard_dark_region - hard_shadow_scores)
        Oi = hard_shadow_scores * hard_shadow_scores
        Oc = 1.0 - hard_dark_region

        pOd = self.LSDForDivergenceBranch(cue) * Od
        pOi = self.LSDForIntersectionBranch(cue) * Oi
        pnD = self.conv(shadow_scores) * Oc
        return pOi, pOd, pnD + pOi + pOd

class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        size = (384,384)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        trans_img = torchvision.transforms.Normalize(mean, std)
        trans_back = torchvision.transforms.Normalize(-mean/std, 1.0/std)
        self.trans_imgs = lambda x: torch.cat([trans_img(x[_]).unsqueeze(0) for _ in range(x.shape[0])], dim=0)
        self.trans_backs = lambda x: torch.cat([trans_back(x[_]).unsqueeze(0) for _ in range(x.shape[0])], dim=0)

        self.gcn = GCN()

        self.DRR = nn.Sequential(
            nn.Conv2d( 3, 64, kernel_size=(7,7), padding=(3,3), bias=False ), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d( 64, 32, kernel_size=(3,3), padding=(1,1), bias=False ), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d( 32, 1, 1, bias=False )
        )
        self.dasa0 = DASA(size)
        self.dasa1 = DASA(size)
        self.dasa2 = DASA(size)
        self.dasa3 = DASA(size)
        self.prediction = nn.Conv2d(4, 1, 1, bias=False)

    def forward(self, x):
        ret = self.gcn(x)
        ss0, ss1, ss2, ss3 = ret["shadow_scores"]
        cue0, cue1, cue2, cue3 = ret["cues"]

        dark_region = self.DRR( x )

        Oi_0, Od_0, dasf0 = self.dasa0(dark_region, ss0, cue0)
        Oi_1, Od_1, dasf1 = self.dasa1(dark_region, ss0, cue1)
        Oi_2, Od_2, dasf2 = self.dasa2(dark_region, ss0, cue2)
        Oi_3, Od_3, dasf3 = self.dasa3(dark_region, ss0, cue3)

        pred = self.prediction(torch.cat([dasf0, dasf1, dasf2, dasf3], dim=1))

        hard_dark_region = torch.sign( torch.relu(dark_region) )
        hard_shadow_scores = torch.sign( torch.relu(ss0) )
        Od = torch.relu(hard_dark_region - hard_shadow_scores)
        Oi = hard_shadow_scores * hard_shadow_scores
        Oc = 1.0 - hard_dark_region

        return {
            "pred": pred,
            "shadow_scores": [ss0, ss1, ss2, ss3],
            "dark_region": dark_region,
            "Oi": [Oi_0, Oi_1, Oi_2, Oi_3],
            "Od": [Od_0, Od_1, Od_2, Od_3],
            "aux_masks": [Oi.detach(), Od.detach(), Oc.detach()]
        }


