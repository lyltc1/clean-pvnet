from torch import nn
import torch
from torch.nn import functional as F
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, \
    estimate_voting_distribution_with_mean
from lib.config import cfg


class Resnet18(nn.Module):
    def __init__(self, num_keypoints):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        backbone = resnet18(fully_conv=True,
                            pretrained=True,
                            output_stride=8,
                            remove_avg_pool_layer=True)

        # replace the last FC layers in origin resnet18 with Conv layers
        backbone.fc = nn.Sequential(
            nn.Conv2d(backbone.inplanes, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.backbone = backbone

        # x2s(64,in/2,in/2), x4s(64,in/4,in/4), x8s(128,in/8,in/8),
        # x16s(256,in/8,in/8),x32s(256,in/8,in/8),xfc(256,in/8,in/8)

        # xfc(256,in/8) + x8s(128,in/8) -> fm(128,in/8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )
        # fm(128,in/8) -> fm(128,in/4)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        # fm(128,in/4) + x4s(64,in/4) -> fm(64,in/4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True)
        )
        # fm(64,in/4) -> fm(64,in/2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # fm(64,in/2) + x2s(64,in/2) -> fm(32,in/2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )
        # fm(32,in/2) -> fm(32,in)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        # in_channels = 32 fm + 3 image
        # out_channels = 1 mask + num_keypoints * 2
        in_channels = 32 + 3
        out_channels = 1 + num_keypoints * 2
        self.convout1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, out_channels, 1, 1)
        )

        # xfc(128,in/8) + x8s(128,in/8) -> fm(128,in/8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )
        # fm(128,in/8) -> fm(128,in/4)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)

        # fm(128,in/4) + x4s(64,in/4) -> fm(64,in/4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True)
        )
        # fm(64,in/4) -> fm(64,in/2)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)

        # fm(64,in/2) + x2s(64,in/2) -> fm(32,in/2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )
        # fm(32,in/2) -> fm(32,in)
        self.up6 = nn.UpsamplingBilinear2d(scale_factor=2)

        # in_channels = 32 fm + 3 rgb + 1 modal mask
        # out_channels = 1 amodal mask
        in_channels = 32 + 3 + 1
        out_channels = 1
        self.convout2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, out_channels, 1, 1)
        )

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        # TODO 根据代码进行修改
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2 // 2, 2)
        mask = output['mask']
        mask[mask > 0.5] = 1.
        mask[mask <= 0.5] = 0.
        amodal_mask = output['amodal_mask']
        amodal_mask[amodal_mask > 0.5] = 1.
        amodal_mask[amodal_mask <= 0.5] = 0.
        if cfg.test.un_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'amodal_mask': amodal_mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'amodal_mask': amodal_mask, 'kpt_2d': kpt_2d})

    def forward(self, x):
        x2s, x4s, x8s, x16s, x32s, xfc = self.backbone(x)

        fm = self.conv1(torch.cat([xfc, x8s], 1))
        fm = self.up1(fm)

        fm = self.conv2(torch.cat([fm, x4s], 1))
        fm = self.up2(fm)

        fm = self.conv3(torch.cat([fm, x2s], 1))
        fm = self.up3(fm)

        # mask + 2 * num_keypoints_map
        out1 = self.convout1(torch.cat([fm, x], 1))
        mask = torch.sigmoid(out1[:, 0:1])  # visible_mask
        ver_pred = out1[:, 1:]

        fm = self.conv4(torch.cat([xfc, x8s], 1))
        fm = self.up4(fm)

        fm = self.conv5(torch.cat([fm, x4s], 1))
        fm = self.up5(fm)

        fm = self.conv6(torch.cat([fm, x2s], 1))
        fm = self.up6(fm)

        # amodal_mask
        out2 = self.convout2(torch.cat([fm, x, mask], 1))
        amodal_mask = torch.sigmoid(out2)  # whole_mask

        ret = {'mask': mask[:, 0], 'amodal_mask': amodal_mask[:, 0], 'vertex': ver_pred}

        if not self.training:
            with torch.no_grad():
                self.decode_keypoint(ret)

        return ret


def get_res_epnet(num_keypoints):
    model = Resnet18(num_keypoints)
    return model
