import torch.nn as nn
from torch.nn import functional as F
from lib.utils import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.mask_crit = torch.nn.functional.binary_cross_entropy

    def forward(self, batch):
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        weight = batch['mask'][:, None].float()
        vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        mask_loss = self.mask_crit(output['mask'], batch['mask'].float())
        scalar_stats.update({'mask_loss': mask_loss})
        loss += mask_loss

        amodal_mask_loss = self.mask_crit(output['amodal_mask'], batch['amodal_mask'].float())
        scalar_stats.update({'amodal_mask_loss': amodal_mask_loss})
        loss += amodal_mask_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
