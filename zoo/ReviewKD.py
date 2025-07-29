from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['hcl', 'build_review_kd2']

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x

class ReviewKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, shapes, out_shapes,
    ):
        super(ReviewKD, self).__init__()
        self.student = student
        self.shapes = shapes
        self.out_shapes = shapes if out_shapes is None else out_shapes

        abfs = nn.ModuleList()

        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        self.abfs = abfs[::-1]
        self.to('cuda')

    def forward(self, x):
        student_features = self.student(x, is_feat=True)
        # student_features[0][-1] = student_features[0][-1].view(student_features[0][-1].shape[0], student_features[0][-1].shape[1], 1, 1)
        student_features[0][-2] = nn.AdaptiveAvgPool2d((1, 1))(student_features[0][-2])
        logit = student_features[1]
        # x = student_features[0][::-1]
        x = student_features[0][:-1][::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)

        return results, logit


def build_review_kd2(model_s, feat_t, feat_s):
    out_shapes = None
    featT = feat_t[1:-1]
    featS = feat_s[1:-1]
    featT[-1] = nn.AdaptiveAvgPool2d((1, 1))(featT[-1])
    featS[-1] = nn.AdaptiveAvgPool2d((1, 1))(featS[-1])
    # featT[-1] = featT[-1].view(featT[-1].shape[0], featT[-1].shape[1], 1, 1)
    # featS[-1] = featS[-1].view(featS[-1].shape[0], featS[-1].shape[1], 1, 1)

    # in_channels = [116, 232, 464, 1024]
    # shapes = [1, 4, 8, 16]
    # out_channels = [64, 128, 256, 256]
    # out_shapes = [1, 8, 16, 32]

    in_channels = [each.shape[1] for each in featS]
    shapes = list(reversed([each.shape[2] for each in featS]))
    out_channels = [each.shape[1] for each in featT]
    out_shapes = list(reversed([each.shape[2] for each in featT]))

    backbone = ReviewKD(
        student=model_s,
        in_channels=in_channels,
        out_channels=out_channels,
        shapes=shapes,
        out_shapes=out_shapes
    )
    return backbone

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

if __name__ == '__main__':
    net = build_review_kd('shufflev2', 100, 'wrn-40-2')

    print('123')
