import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):  # 本来factor=32
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=GELU(), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(EMA(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class MRCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act=GELU()):
        super(MRCAB, self).__init__()
        up_body=[RCAB(conv, n_feat, kernel_size, act=act
                           )for _ in range(30)]
        self.body = nn.Sequential(*up_body)
        self.act = nn.ReLU(True)
        self.conv0 = nn.Conv2d(n_feat, n_feat, 1, 1, 0)
        self.conv1 = nn.Conv2d(n_feat, n_feat, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, 1, 0),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        )

    def forward(self, x):
        x1 = x.clone()
        x1 = self.conv0(x1)
        a1 = self.conv1(x1)
        a2 = self.conv2(x1)
        a3 = self.conv3(x1)
        a = a1 + a2 + a3
        a = self.act(a)
        res = self.body(x)
        res = a + res + x1

        return res

