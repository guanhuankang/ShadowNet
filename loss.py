import os
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

from misc import log, metric, AvgMeter
from config import Config

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.trans_img = torchvision.transforms.Normalize(mean, std)
        self.trans_back = torchvision.transforms.Normalize(-mean/std, 1.0/std)
        self.trans_imgs = lambda x: torch.cat([self.trans_img(x[_]).unsqueeze(0) for _ in range(x.shape[0])], dim=0)
        self.trans_backs = lambda x: torch.cat([self.trans_back(x[_]).unsqueeze(0) for _ in range(x.shape[0])], dim=0)

        self.epo = torch.tensor(1e-9).cuda()
        self.tot_iteration = Config().data()["max_iter"]
        self.shadow_recall_factor = Config().shadow_recall_factor

        self.pl = AvgMeter()
        self.bl = AvgMeter()
        self.ml = AvgMeter()
        self.fl = AvgMeter()
        self.sm = AvgMeter()

    def forward(self, optimizer, net, data, index):
        img, gt = data[0].cuda(), data[1].cuda()
        gt = gt.gt(0.5).float()

        bs = img.shape[0]

        trans_img = torch.stack( [self.trans_img(img[_]) for _ in range(bs) ], dim = 0 )
        out = net( trans_img )

        # return {
        #     "pred": pred,
        #     "shadow_scores": [ss0, ss1, ss2, ss3],
        #     "dark_region": dark_region,
        #     "Oi": [Oi_0, Oi_1, Oi_2, Oi_3],
        #     "Od": [Od_0, Od_1, Od_2, Od_3],
        #     "aux_masks": [Oi, Od, Oc]
        # }
        # final_loss = self.bceloss(out["pred"], gt, self.shadow_recall_factor) ## 1.10 for SBU & 1.00 for ISTD
        # gcn_loss = self.predict_loss(out["shadow_scores"], gt)
        # ddr_loss = self.bceloss(out["dark_region"], gt)
        # dasa_i_loss = self.mask_loss(out["Oi"], gt, out['aux_masks'][0])
        # dasa_d_loss = self.mask_loss(out["Od"], gt, out['aux_masks'][1])

        final_loss = F.binary_cross_entropy_with_logits(out["pred"], gt)
        gcn_loss = sum([F.binary_cross_entropy_with_logits(x, gt) for x in out["shadow_scores"]])
        ddr_loss = F.binary_cross_entropy_with_logits(out["dark_region"], gt)

        Oi, Od = out['aux_masks'][0], out['aux_masks'][1]
        dasa_i_loss = sum([torch.sum(F.binary_cross_entropy_with_logits(x, gt, reduction="none") * Oi) / (Oi.sum() + 1e-6) for x in out["Oi"]])
        dasa_d_loss = sum([torch.sum(F.binary_cross_entropy_with_logits(x, gt, reduction="none") * Od) / (Od.sum() + 1e-6) for x in out["Od"]])

        loss = final_loss + gcn_loss + ddr_loss + dasa_i_loss +dasa_d_loss

        self.fl.update(final_loss.item())
        self.pl.update(gcn_loss.item())
        self.bl.update(ddr_loss.item())
        self.ml.update(dasa_i_loss.item())
        self.sm.update(dasa_d_loss.item())

        ret = loss.item()
        log( "iter:{:4d}, lr:{:.5f}, loss:{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(
            index,
            optimizer.param_groups[0]["lr"],
            self.fl.avg,
            self.pl.avg,
            self.bl.avg,
            self.ml.avg,
            self.sm.avg
        ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index%10==9:
            log(str(metric( nn.Sigmoid()(out["pred"]), gt)))
        return ret
    
    def bceloss(self, m, gt, factor=1.0):
        gt = F.interpolate(gt.data, size=m.shape[-2::])
        gt = gt.gt(0.5).float()
        
        neg_cnt = (1.0-gt).sum()*1.0
        pos_cnt = gt.sum()*1.0
        beta = neg_cnt/ (self.epo+pos_cnt) * factor
        beta_back = pos_cnt/(pos_cnt+neg_cnt)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=beta)
        return beta_back * criterion(m, gt)

    def predict_loss(self, maps, gt):
        loss_pos = self.bceloss(maps[0], gt)
        T = len(maps)
        for i in range(1,T):
            loss_pos += self.bceloss(maps[i], gt)
        return loss_pos
    
    def single(self, m, gt, mask, factor=1.0):
        gt = gt.gt(0.5).float()
        sbm = torch.sign(torch.relu(mask))

        neg_cnt = ((1.0-gt)*sbm.data).sum()*1.0
        pos_cnt = (gt*sbm.data).sum()*1.0
        beta = neg_cnt/ (self.epo+pos_cnt) * factor
        beta_back = pos_cnt/(pos_cnt+neg_cnt)

        neg_cnt = (1.0-gt).sum()*1.0
        pos_cnt = gt.sum()*1.0
        beta += neg_cnt/ (self.epo+pos_cnt)
        beta_back += pos_cnt/(pos_cnt+neg_cnt)

        lo = beta *sbm.data * gt * F.binary_cross_entropy_with_logits( m, gt, reduce=False ) \
            + sbm.data * (1.0-gt) * F.binary_cross_entropy_with_logits( m, gt, reduce=False )
        return lo.mean() * beta_back
        
    def mask_loss(self, ms, gt, mask):
        lo = self.single(ms[0], gt, mask)
        T = len(ms)
        for i in range(1, T):
            lo += self.single(ms[i], gt, mask)
        return lo
