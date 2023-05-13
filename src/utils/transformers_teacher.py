from ast import Param
from curses.ascii import VT
from tokenize import group
from warnings import WarningMessage
import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init, Conv2d, \
    Sequential, ReLU, Hardtanh, ReLU6, LayerNorm, Softmax, LeakyReLU, PReLU
import torch.nn.functional as F
import torch.nn as nn
from .stochastic_depth import DropPath
import math
import pdb
from src.utils.rmsnorm import GatedRMSNorm
from src.utils.sparsemax import Sparsemax
from src.utils.activation import Learnable_Relu
import numpy as np


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)  # try to extend
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)  # try to extend
        self.proj_drop = Dropout(projection_dropout)

        ## addcode-start ##
        self.qkv_expand = Sequential(
            Linear(dim, dim * 6, bias=False), 
            Linear(dim * 6, dim * 3, bias=False)
            # immediate nonlinear function (relu)
        )
        self.proj_expand = Sequential(
            Linear(dim, 4 * dim),
            Linear(4 * dim, 1 * dim)
            # 1 -> 4 -> 1
            # immediate nonlinear function (relu)
        )
        self.dw_conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3,\
            stride=1, padding=1, groups=256, bias=False)
        self.W = Parameter(torch.zeros(size=(1, 256, 64, 64)), requires_grad=True)  # (batch_size, channel, HW, HW) 
        # batch_size=128 -> cuda out of memory. Here set 1 and broadcast to batch_size (less parameters)
        init.xavier_uniform_(self.W.data, gain=1.414)
        self.kernel_s = Parameter(torch.zeros(self.num_heads, 1, 3, 3),  requires_grad=True)  
        # follow the rule: dwconv -> (out_channels, 1, K, K)
        self.kernel_o = Parameter(torch.zeros(1, 1, 3, 3), requires_grad=True)
        
        # self.hardtanh = Hardtanh(min_val= 0.5, max_val=1.5)
        # self.gamma = Parameter(torch.randn((1, num_heads, 1, 1)))
        # self.relu6 = ReLU6()
        # self.ln = LayerNorm(256)
        # self.rmsnorm1 = GatedRMSNorm(256)
        # self.rmsnorm2 = GatedRMSNorm(64)
        # self.sparsemax = Sparsemax()
        # self.leakyrelu = LeakyReLU()
        # self.prelu = PReLU()
        # self.learnable_relu = Learnable_Relu()
        
        self.relu = ReLU()
        self.eps = 1e-8  # 1e-6 for cifar-10 & 1e-7 for cifar-100
        # self.alpha = Parameter(torch.FloatTensor(1), requires_grad=False)
        # self.alpha = Parameter(torch.ones(1, self.num_heads, 1, 1), requires_grad=True)
        # self.alpha.data.fill_(1.0)
        ## addcode-end ##

    def forward(self, x):
        B, N, C = x.shape  # x: (B, HW + 1, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q/k/v: (B, #heads, HW + 1, C // #heads)

        ## original dot product attention ##
        attn_ = (q @ k.transpose(-2, -1))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # attn: (B, #heads, HW + 1, HW + 1)

        # attn_scaling = attn / attn.size(3)
        # attn = self.relu(attn)
        # attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + self.eps)
        # # print('Execute ReLU + Norm') if self.alpha == 1 else print('Execute Scaling Attention')
        # attn = self.alpha * attn + (1 - self.alpha) * attn_scaling

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)  # x: (B, HW + 1, C)

        ## addcode-start: XNorm Attention ##
        # kv = k.transpose(-2, -1) @ v  # kv: (B, #heads, C // #heads, C // #heads)
        # kv_normed = self.XNorm(kv, self.gamma)  # kv_normed: (B, #heads, C // #heads, C // #heads)
        # q_normed = self.XNorm(q, self.gamma)  # q_normed: (B, #heads, HW + 1, C // #heads)
        # x = (q_normed @ kv_normed).transpose(1, 2).reshape(B, N, C)  # x: (B, HW + 1, C)
        ## addcode-end ##

        # addcode-start: Hydra Attention (cosine similarity) ##
        # q/k/v: (B, HW+1, #heads), #heads = #channels
        # k = k.squeeze(-1).transpose(-2, -1)
        # q = q.squeeze(-1).transpose(-2, -1)
        # v = v.squeeze(-1).transpose(-2, -1)
        # q = q / q.norm(dim=-1, keepdim=True)
        # k = k / k.norm(dim=-1, keepdim=True)
        # kv = (k * v).sum(dim=-2, keepdim=True)
        # x = q * kv  # x: (B, HW + 1, C)
        # addcode-end ##

        ## addcode-start ##
        # v_t = v.transpose(-2, -1).flatten(1, 2)
        # clf_token = v_t[:, :, 0]
        # v_t_dropclf = v_t[:, :, 1:].reshape(-1, C, int(math.sqrt(N)), int(math.sqrt(N)))

        ## Operation1: Depth-wise convolution
        # v_conv = self.dw_conv(v_t_dropclf)
        # v_conv_flatten = v_conv.flatten(-2, -1)
        # conv_output = torch.cat((v_conv_flatten, clf_token.unsqueeze(2)), dim=2).transpose(-1, -2)
        # x = x + conv_output

        ## Operation2: compute DWConv with a trainable W
        # out = (self.W @ (v_t[:, :, 1:].unsqueeze(3))).squeeze(3)
        # out_cat = torch.cat((out, clf_token.unsqueeze(2)), dim=2).transpose(-1, -2)
        # # x = x + out_cat

        ## Operation3: Weight-sharing depth-wise convolution for each head
        # kernel_m = torch.repeat_interleave(self.kernel_s, C // self.num_heads, dim=0)
        # out_list = list()
        # for h in range(self.num_heads):            
        #     kernel = kernel_m[h * C // self.num_heads:(h + 1) * (C // self.num_heads)]
        #     v_h = v[:, h, :, :].transpose(-1, -2)  # v_h: (B, C // #heads, HW + 1)
        #     clf_token = v_h[:, :, 0]
        #     v_h_dropclf = v_h[:, :, 1:].reshape(-1, C // self.num_heads, int(math.sqrt(N)), int(math.sqrt(N)))
        #     dwconv_in_head = F.conv2d(input=v_h_dropclf, weight=kernel, bias=None, stride=1, padding=1, groups=C//self.num_heads)
        #     dwconv_in_head_flatten = dwconv_in_head.flatten(-2, -1)
        #     output_in_head = torch.cat((dwconv_in_head_flatten, clf_token.unsqueeze(2)), dim=2).transpose(-1, -2)
        #     out_list.append(output_in_head)
        # output = torch.cat(out_list, dim=-1)  # now have the same dimension with attn output
        # x = x + output

        ## Operation4: Weight-sharing depth-wise convolution for all channels
        # kernel_m = torch.repeat_interleave(self.kernel_o, C, dim=0)
        # dwconv = F.conv2d(input=v_t_dropclf, weight=kernel_m, bias=None, stride=1, padding=1, groups=C)
        # dwconv_flatten = dwconv.flatten(-2, -1)
        # output = torch.cat((dwconv_flatten, clf_token.unsqueeze(2)), dim=2).transpose(-1, -2)
        # x = x + output

        ## addcode-ends ##

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    ## addcode-start: construct the weight_matrix ##
    def get_weight_matrix(self, H, W, batch_size, channel, dw_kernel):
        id_map = torch.arange(H * W).reshape(H, W)
        id_map_repeat = id_map.repeat(batch_size, channel, 1, 1)
        id_map_pad = F.pad(input=id_map_repeat, pad=[1, 1, 1, 1], mode="constant", value=-1)
        row = 0
        weight_mx = torch.zeros(batch_size, channel, H * W, H * W)
        for i in range(W):
            for j in range(H):
                input_conv = id_map_pad[0, 0, i:i+3, j:j+3]
                loc_on_weight = input_conv[torch.where(input_conv != -1)]
                for k in loc_on_weight:
                    weight_mx[:, :, row, k] = dw_kernel[:, 0, torch.where(input_conv == k)[0], torch.where(input_conv == k)[1]]\
                        .squeeze(1).repeat(batch_size, 1)
                row += 1
        return weight_mx
    ## addcode-end ##

    ## addcode-start: XNorm ##
    def XNorm(self, x, gamma):
        norm_tensor=torch.norm(x, 2, -1, True)
        return x * gamma / norm_tensor
    ## addcode-end ##


class MaskedAttention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

        ## addcode-start: expand the fc layer (linear1) ##
        ratio_expand = int(2)
        self.linear_expand =  Sequential(
            Linear(d_model, ratio_expand * d_model),
            Linear(ratio_expand * d_model, dim_feedforward)
        )
        ## addcode-end ##

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # print(self.pre_norm(src).shape)
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)

        ## addcode-start ##
        is_expand = False
        is_inference = False
        if is_expand and not is_inference:  # training time
            src2 = self.linear2(self.dropout1(self.activation(self.linear_expand(src))))
        elif is_expand and is_inference:  # infernece time
            tmp = self.fuse_fc(self.linear_expand[0], self.linear_expand[1])
            self.linear1.weight.data, self.linear1.bias.data = tmp['weight'], tmp['bias']
            src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
            # may occur this bug: "Attempting to unscale FP16 gradients."
            # solution: set torch.cuda.amp/grad_scaler.py/_unscale_grads_() allow_fp16=True
        ## addcode-end ##
        if not is_expand:
            src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))  # original line

        src = src + self.drop_path(self.dropout2(src2))
        return src
    
    ## addcode-start: fuse two fc layers ##
    def fuse_fc(self, fc1: Linear, fc2: Linear):
        """
        Fuse equivalent weight from two fc layers
        :param dim_in: m
        :param dim_out: n
        :param dim_immediate: p
        :param s_1: p * m
        :param s_2: n * p
        :return: fused weight m * n and bias
        """
        if isinstance(fc1, Linear):
            w_s_1 = fc1.weight
            b_s_1 = fc1.bias
        else:
            w_s_1 = fc1['weight']
            b_s_1 = fc1['bias']

        if isinstance(fc2, Linear):
            w_s_2 = fc2.weight
            b_s_2 = fc2.bias
        else:
            w_s_2 = fc2['weight']
            b_s_2 = fc2['bias']

        if b_s_1 is not None and b_s_2 is not None:
            new_bias = torch.matmul(w_s_2, b_s_1) + b_s_2
        elif b_s_1 is None:
            new_bias = b_s_2  # without bias1
        else:
            new_bias = None

        new_weight = torch.matmul(w_s_2, w_s_1)

        return {'weight': new_weight, 'bias': new_bias}
        ## addcode-end ##


class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead,
                                         attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier_teacher(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

        ## addcode-start: expand the fc layer ##
        ratio_expand = int(2)
        self.fc_expand =  Sequential(
            Linear(embedding_dim, ratio_expand * embedding_dim),
            Linear(ratio_expand * embedding_dim, num_classes)
        )
        ## addcode-end ##

    def forward(self, x):
        # print('x:', x.shape)
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
        
        features = []

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)  # (128, 256, 256)

        x = self.norm(x)
        
        features.append(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]  # fetech the classification token

        ## addcode-start ##
        is_expand = False
        is_inference = False
        if is_expand and not is_inference:  # training time
            x = self.fc_expand(x)
        elif is_expand and is_inference:  # infernece time
            tmp = self.fuse_fc(self.fc_expand[0], self.fc_expand[1])
            self.fc.weight.data, self.fc.bias.data = tmp['weight'], tmp['bias']
            x = self.fc(x)
            # may occur this bug: "Attempting to unscale FP16 gradients."
            # solution: set torch.cuda.amp/grad_scaler.py/_unscale_grads_() allow_fp16=True
        ## addcode-end ##
        if not is_expand:
            x = self.fc(x)  # original line

        return x, features

    ## addcode-start: fuse two fc layers ##
    def fuse_fc(self, fc1: Linear, fc2: Linear):
        """
        Fuse equivalent weight from two fc layers
        :param dim_in: m
        :param dim_out: n
        :param dim_immediate: p
        :param s_1: p * m
        :param s_2: n * p
        :return: fused weight m * n and bias
        """
        if isinstance(fc1, Linear):
            w_s_1 = fc1.weight
            b_s_1 = fc1.bias
        else:
            w_s_1 = fc1['weight']
            b_s_1 = fc1['bias']

        if isinstance(fc2, Linear):
            w_s_2 = fc2.weight
            b_s_2 = fc2.bias
        else:
            w_s_2 = fc2['weight']
            b_s_2 = fc2['bias']

        if b_s_1 is not None and b_s_2 is not None:
            new_bias = torch.matmul(w_s_2, b_s_1) + b_s_2
        elif b_s_1 is None:
            new_bias = b_s_2  # without bias1
        else:
            new_bias = None

        new_weight = torch.matmul(w_s_2, w_s_1)

        return {'weight': new_weight, 'bias': new_bias}
    ## addcode-end ##

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MaskedTransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='sine',
                 seq_len=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert seq_len is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len,
                                                                          embedding_dim,
                                                                          padding_idx=True),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            MaskedTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask=None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
                mask = (mask > 0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim, padding_idx=False):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
        return pe
