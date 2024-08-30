from .Swin import SwinTransformer
from torch import nn, Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
# from modules import DeformConv
import copy
from timm.models.layers import trunc_normal_
import numpy as np
import torch.nn.init as init
# zeros_ = torch.nn.init.constant_(val=0.0)

def to_2tuple(x):
    """to_2tuple"""
    return tuple([x] * 2)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = torch.tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """forward"""
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):
    """Identity"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        """forward"""
        return input


class Mlp(nn.Module):
    """Mlp"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """forward"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def masked_fill(x, mask, value):
    """masked_fill"""
    y = torch.full(x.shape, value, x.dtype)
    return torch.where(mask, y, x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            mask = mask.to(attn.device)
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """Block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        """forward"""
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        epsilon=1e-6,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = (
            norm_layer(embed_dim, eps=epsilon) if norm_layer else Identity()
        )

    def forward(self, x):
        """forward"""
        # B, C, H, W = x.shape
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), "Input image size ({}*{}) doesn't match model ({}*{}).".format(
            H, W, self.img_size[0], self.img_size[1]
        )

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).permute((0, 2, 1))
        x = self.norm(x)
        return x

# def get_sinusoid_encoding(n_position, d_hid):
#     ''' Sinusoid position encoding table '''
#
#     def get_position_angle_vec(position):
#         return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
#
#     sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
#
#     return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

# def conv3x3_bn_relu_2(in_planes, out_planes, stride=2):
#     return nn.Sequential(
#             conv3x3(in_planes, out_planes, stride),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU(inplace=True),
#             )

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



#model
class TMSOD(nn.Module):
    def __init__(self):
        super(TMSOD, self).__init__()
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.MHSA4 = GMSA_ini(d_model=1024)
        self.MHSA3 = GMSA_ini(d_model=512)
        self.MHSA2 = GMSA_ini(d_model=256)
        self.MHSA1 = GMSA_ini(d_model=128)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        # self.conv4 =

        self.convr4 = conv3x3_bn_relu(1024, 512)
        self.convr3 = conv3x3_bn_relu(512, 256)
        self.convr2 = conv3x3_bn_relu(256, 128)
        self.convr1 = conv3x3_bn_relu(128, 64)
        self.conv_dim = conv3x3(64, 1)

    def forward(self, rgb):
        #rgb fea
        fr = self.rgb_swin(rgb)  # [0-3]

        flatten4 = fr[3].flatten(2).transpose(1, 2)# B HW dim
        flatten3 = fr[2].flatten(2).transpose(1, 2)
        flatten2 = fr[1].flatten(2).transpose(1, 2)
        # flatten1 = fr[0].flatten(2).transpose(1, 2)

        r4 = self.MHSA4(flatten4, flatten4).view(flatten4.shape[0], int(np.sqrt(flatten4.shape[1])), int(np.sqrt(flatten4.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        r3 = self.MHSA3(flatten3, flatten3).view(flatten3.shape[0], int(np.sqrt(flatten3.shape[1])), int(np.sqrt(flatten3.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        r2 = self.MHSA2(flatten2, flatten2).view(flatten2.shape[0], int(np.sqrt(flatten2.shape[1])), int(np.sqrt(flatten2.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        # r1 = self.MHSA1(flatten1, flatten1).view(flatten1.shape[0], int(np.sqrt(flatten1.shape[1])), int(np.sqrt(flatten1.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        r1 = fr[0]

        r4 = self.convr4(self.up2(r4))
        r3 = self.convr3(self.up2(r3 + r4))
        r2 = self.convr2(self.up2(r2 + r3))
        r1 = self.convr1(r1 + r2)
        out = self.up4(r1)
        out = self.conv_dim(out)

        return out

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        # self.t_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        # print(f"Depth SwinTransformer loading pre_model ${pre_model}")

class GMSA_ini(nn.Module):
    def __init__(self, d_model=256, num_layers=4, decoder_layer=None):
        super(GMSA_ini, self).__init__()
        if decoder_layer is None:
            decoder_layer = GMSA_layer_ini(d_model=d_model, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)
    def forward(self, fr, ft):
        # fr = fr.flatten(2).transpose(1, 2)  # b hw c
        # ft = ft.flatten(2).transpose(1, 2)
        output = fr
        for layer in self.layers:
            output = layer(output, ft)
        return output
class GMSA_layer_ini(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GMSA_layer_ini, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()
    def forward(self, fr, ft, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):


        fr2 = self.multihead_attn(query=self.with_pos_embed(fr, query_pos).transpose(0, 1),#hw b c
                                   key=self.with_pos_embed(ft, pos).transpose(0, 1),
                                   value=ft.transpose(0, 1))[0].transpose(0, 1)#b hw c
        fr = fr + self.dropout2(fr2)
        fr = self.norm2(fr)

        fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))  #FFN
        fr = fr + self.dropout3(fr2)
        fr = self.norm3(fr)
        # print(fr.shape)
        return fr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

#gated MSA
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# gated MSA layer
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class get_aligned_feat(nn.Module):
    def __init__(self, inC, outC):
        super(get_aligned_feat, self).__init__()
        self.deformConv1 = defomableConv(inC=inC*2, outC=outC)
        self.deformConv2 = defomableConv(inC=inC, outC=outC)
        self.deformConv3 = defomableConv(inC=inC, outC=outC)
        self.deformConv4 = defomableConv_offset(inC=inC, outC=outC)

    def forward(self, fr, ft):
        cat_feat = torch.cat((fr, ft), dim=1)
        feat1 = self.deformConv1(cat_feat)
        feat2 = self.deformConv2(feat1)
        feat3 = self.deformConv3(feat2)
        aligned_feat = self.deformConv4(feat3, ft)
        return aligned_feat

class defomableConv(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 4):
        super(defomableConv, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, x):
        offset = self.offset(x)
        out = self.deform(x, offset)
        return out

class defomableConv_offset(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 2):
        super(defomableConv_offset, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, feta3, x):
        offset = self.offset(feta3)
        out = self.deform(x, offset)
        return out