import torch

from ..spa_spe import *
from einops import rearrange


# stem
class stemBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(stemBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv3d(1, 12, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(12)
        self.conv3 = nn.Conv3d(12, 12, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3), bias=False)
        self.bn3 = nn.BatchNorm3d(12)
        self.conv4 = nn.Conv3d(12, 1, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=False)
        self.bn4 = nn.BatchNorm3d(1)
        self.conv5 = nn.Conv2d(256, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x1 = x.unsqueeze(1)
        x1 = x1.transpose(-1, 2)
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = x1.transpose(-1, 2)
        x1 = x1.squeeze(1)
        x = F.relu(self.bn5(self.conv5(x1)))
        return x


class neigh_embed(nn.Module):
    def __init__(self, channel, neigh_number):
        super(neigh_embed, self).__init__()

        self.neigh_Branch = nn.Sequential()
        self.neigh_number = neigh_number
        for i in range(channel):
            self.neigh_Branch.add_module('neigh_Branch' + str(i), nn.Conv2d(neigh_number, 1, kernel_size=(1, 1), stride=1))

    def forward(self, x):
        b, c, w, h = x.shape
        start = int((self.neigh_number-1)/2)  # 3 1
        end = int(c-1-start)  # c-1
        for i in range(c):
            self_c = x[:, i, :, :]
            self_c = self_c.unsqueeze(1)
            if i == 0:
                A = self_c+self.neigh_Branch[i](x[:,i:i+self.neigh_number, :, :])  # [64 1 21 1]
            if i > 0:
                if i < start:
                    B = self_c + self.neigh_Branch[i](x[:, 0:self.neigh_number, :, :])  # [64 1 21 1]
                if i >= start and  i<= end:
                    B = self_c + self.neigh_Branch[i](x[:, (i-start):(i-start + self.neigh_number), :, :])  # [64 1 21 1]
                if i > end:
                    B = self_c + self.neigh_Branch[i](x[:, c-self.neigh_number:c, :, :])  # [64 1 21 1]
                A = torch.cat((A, B), 1)
        return A


class Embedding(nn.Module):
    def __init__(self, in_channel, embed_dim, out_channel):
        super().__init__()

        self.pre_norm = nn.LayerNorm(in_channel)
        self.conv1 = nn.Conv2d(in_channel, embed_dim, 3, 1, 1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.neigh_conv = neigh_embed(channel=embed_dim, neigh_number=3)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(embed_dim, out_channel, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.pre_norm(x)
        x = x.permute(0, 2, 1).view(B, -1, H, W)
        x1 = self.conv1(x)
        x1 = x1.flatten(2).permute(0, 2, 1)
        x1 = self.act1(self.norm1(x1))
        x1 = x1.permute(0, 2, 1).view(B, -1, H, W)
        x2 = self.neigh_conv(x1)
        x3 = self.act2(self.batch_norm(x2))
        x4 = self.conv2(x3)  # B, C, H, W

        return x4


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


# stage-1
class MogaBlock_pre(nn.Module):
    def __init__(self, dims, neigbor_dims, embed_dims, spe_ratio, attn_channel_split, drop_rate=0., drop_path_rate=0.,
                 act_type='GELU', norm_type='BN', init_value=1e-5, attn_dw_dilation=[1, 2, 3],
                 attn_act_type='SiLU', attn_force_fp32=False,):
        super(MogaBlock_pre, self).__init__()

        self.patch_embed = Embedding(in_channel=dims, embed_dim=neigbor_dims, out_channel=embed_dims)
        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(embed_dims, attn_dw_dilation=attn_dw_dilation,
                                               attn_channel_split=attn_channel_split, attn_act_type=attn_act_type,
                                               attn_force_fp32=attn_force_fp32)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * spe_ratio)
        # DWConv + Channel Aggregation FFN
        self.mlp = ChannelAggregationFFN(embed_dims=embed_dims, feedforward_channels=mlp_hidden_dim, act_type=act_type,
                                         ffn_drop=drop_rate)

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)

        return x


# stage-2/3/4......
class MogaBlock(nn.Module):
    def __init__(self, in_dims, out_dims, spe_ratio, attn_channel_split, drop_rate=0., drop_path_rate=0.,
                 act_type='GELU', norm_type='BN', init_value=1e-5, attn_dw_dilation=[1, 2, 3],
                attn_act_type='SiLU', attn_force_fp32=False,):
        super(MogaBlock, self).__init__()

        self.sampling = nn.Conv2d(in_dims, out_dims, 1, 1)
        self.norm1 = build_norm_layer(norm_type, out_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(out_dims, attn_dw_dilation=attn_dw_dilation,
                                               attn_channel_split=attn_channel_split, attn_act_type=attn_act_type,
                                               attn_force_fp32=attn_force_fp32)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_type, out_dims)

        # channel MLP
        mlp_hidden_dim = int(out_dims * spe_ratio)
        # DWConv + Channel Aggregation FFN
        self.mlp = ChannelAggregationFFN(embed_dims=out_dims, feedforward_channels=mlp_hidden_dim, act_type=act_type,
                                         ffn_drop=drop_rate)

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, out_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, out_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.sampling(x)
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)

        return x


class BCA(nn.Module):
    def __init__(self, c, num_heads):
        super(BCA, self).__init__()
        super().__init__()

        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.norm_d = LayerNorm2d(c)
        self.norm_g = LayerNorm2d(c)
        self.d_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.d_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_d, x_g):
        b, c, h, w = x_d.shape

        x_d = self.norm_d(x_d)
        Q_d = self.d_proj1(x_d)  # B, C, H, W
        x_g = self.norm_d(x_g)
        Q_g_T = self.g_proj1(x_g)  # B, C, H, W

        V_d = self.d_proj2(x_d)  # B, C, H, W
        V_g = self.g_proj2(x_g)  # B, C, H, W

        Q_d = rearrange(Q_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        Q_g_T = rearrange(Q_g_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_d = rearrange(V_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_g = rearrange(V_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q_d = torch.nn.functional.normalize(Q_d, dim=-1)
        Q_g_T = torch.nn.functional.normalize(Q_g_T, dim=-1)

        # (B, head, c, hw) x (B, head, hw, c) -> (B, head, c, c)
        attention = (Q_d @ Q_g_T.transpose(-2, -1)) * self.scale

        F_g2d = torch.matmul(torch.softmax(attention, dim=-1), V_g)  # B, head, c, hw
        F_d2g = torch.matmul(torch.softmax(attention, dim=-1), V_d)  # B, head, c, hw

        # scale
        F_g2d = rearrange(F_g2d, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_d2g = rearrange(F_d2g, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out1 = x_d + F_g2d * self.beta
        out2 = x_g + F_d2g * self.gamma
        out = torch.cat((out1, out2), dim=1)

        return out


class MyNet(nn.Module):
    def __init__(self, band, num_classes, num_heads, spe_ratio, attn_channel_split, dims=[128, 96, 64, 32],):
        super().__init__()

        self.stem = stemBlock(in_channel=band, out_channel=dims[0])

        self.stage1 = MogaBlock_pre(dims=dims[0], neigbor_dims=dims[3], embed_dims=dims[1],
                                    spe_ratio=spe_ratio, attn_channel_split=attn_channel_split)
        self.stage2 = MogaBlock(in_dims=dims[1], out_dims=dims[2],
                                spe_ratio=spe_ratio, attn_channel_split=attn_channel_split)
        self.stage3 = MogaBlock(in_dims=dims[2], out_dims=dims[3],
                                spe_ratio=spe_ratio, attn_channel_split=attn_channel_split)

        self.fusion1 = BCA(c=dims[3], num_heads=num_heads)
        self.fusion2 = BCA(c=dims[2], num_heads=num_heads)

        self.conv1 = nn.Conv2d(dims[2], dims[3], 1, 1)
        self.conv2 = nn.Conv2d(dims[1], dims[2], 1, 1)

        # ------------------------------------------------Classifier head-----------------------------------------------
        self.pool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(0.)
        self.fc = nn.Linear(dims[0], num_classes, bias=False)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.stem(x)

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        y1 = self.conv1(x2)  # B, 32, H, W
        y2 = self.fusion1(x3, y1)  # B, 64, H, W
        y3 = self.conv2(x1)  # B, 64, H, W
        y4 = self.fusion2(y2, y3)  # B, 128,H,W

        # for image classification
        out = self.pool(self.dropout(y4)).view(-1, y4.shape[1])
        out = self.fc(out)

        return out


def Moga(dataset):
    model = None
    if dataset == 'SV':  # batchsize=96, lr=0.005
        model = MyNet(band=204, num_classes=16, num_heads=2, spe_ratio=2, attn_channel_split=[1, 6, 9])
    elif dataset == 'PU':  # batchsize=64, lr=0.008
        model = MyNet(band=103, num_classes=9, num_heads=4, spe_ratio=2, attn_channel_split=[0, 1, 1])
    elif dataset == 'HUST2013':  # batchsize=64, lr=0.005
        model = MyNet(band=144, num_classes=15, num_heads=1, spe_ratio=4, attn_channel_split=[1, 3, 4])
    elif dataset == 'WHU_LK':  # batchsize=64, lr=0.005
        model = MyNet(band=270, num_classes=9, num_heads=1, spe_ratio=2, attn_channel_split=[1, 3, 4])
    elif dataset == 'WHU_HC':  # batchsize=64, lr=0.005
        model = MyNet(band=274, num_classes=16, num_heads=1, spe_ratio=2, attn_channel_split=[1, 3, 4])
    return model


if __name__ == "__main__":
    t = torch.randn(size=(64, 103, 13, 13))
    print("input shape:", t.shape)
    net = Moga(dataset='PU')
    net.eval()
    print(net)
    print("output shape:", net(t).shape)
