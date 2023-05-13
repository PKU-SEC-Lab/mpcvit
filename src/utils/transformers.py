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
  
        ## Exploration of different attention variants
        # self.hardtanh = Hardtanh(min_val= 0.5, max_val=1.5)
        # self.gamma_xnorm = Parameter(torch.randn((1, num_heads, 1, 1)))
        # self.relu6 = ReLU6()
        # self.ln = LayerNorm(256)
        # self.rmsnorm1 = GatedRMSNorm(256)
        # self.rmsnorm2 = GatedRMSNorm(64)
        # self.sparsemax = Sparsemax()
        # self.leakyrelu = LeakyReLU()
        # self.prelu = PReLU()
        # self.learnable_relu = Learnable_Relu()
        
        self.relu = ReLU()
        self.eps = 1e-5
        # self.alpha = Parameter(torch.FloatTensor(1), requires_grad=True)  # layer-wise search
        self.alpha = Parameter(torch.ones(1, self.num_heads, 1, 1), requires_grad=True)  # head-wise search
        # self.alpha = Parameter(torch.ones(1, 1, 65, 1), requires_grad=True)  # row-wise search
        self.alpha.data.fill_(1.0)
        # self.kk = 77
        # self.E = Parameter(torch.zeros(1, 1, self.kk, 65))
        # init.xavier_uniform_(self.E.data, gain=1.414)
        # self.t1 = Linear(1, 64)
        # self.t2 = Linear(64, 128)
        # self.t3 = Linear(128, 197)

    def forward(self, x):
        B, N, C = x.shape  # x: (B, HW + 1, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q/k/v: (B, #heads, HW + 1, C // #heads)

        '''
        # LinFormer
        k = self.E @ k
        v = self.E @ v
        '''

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = (attn ** 2) / (torch.sum((attn ** 2), dim=-1, keepdim=True) + self.eps)  # mpcformer
        # attn = attn.softmax(dim=-1)  # attn: (B, #heads, HW + 1, HW + 1) - Vallina Attention
        
        # attn[torch.where(attn < 0)] = 0
        # attn[torch.where(attn > 0)] = attn[torch.where(attn > 0)] * attn[torch.where(attn > 0)]
        # attn = torch.where(attn < 0, 0, attn ** 2)

        scalattn = attn / attn.size(3)  # scaling attention
        attn = self.relu(attn)
        attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + self.eps)  # ReLUSoftmax attention

        # attn = attn * self.t3(self.t2(self.t1(torch.sum(self.relu((attn / 2 + 1) ** 3), dim=-1, keepdim=True))))

        # print('Execute ReLUSoftmax') if self.alpha == 1 else print('Execute Scaling Attention')
        attn = self.alpha * attn + (1 - self.alpha) * scalattn  # weighted-sum for arch searching

        # self.alpha.data = torch.clamp(self.alpha.data, min=0, max=1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)  # x: (B, HW + 1, C)

        use_linear_scalattn = False  # reduce the computation of scalattn
        if use_linear_scalattn:
            x_scalattn = (q / math.sqrt(N)) @ (k.transpose(-2, -1) / math.sqrt(N) @ v) * self.scale
            x_scalattn = x_scalattn.transpose(1, 2).reshape(B, N, C)
            attn_relu = self.relu((q @ k.transpose(-2, -1)) * self.scale)
            attn_relu = attn_relu / (torch.sum(attn_relu, dim=-1, keepdim=True) + self.eps)
            attn_relu = self.attn_drop(attn_relu)
            x_attn_relu = (attn_relu @ v).transpose(1, 2).reshape(B, N, C)
            if self.alpha.shape[1] == self.num_heads:
                self.alpha.data = torch.repeat_interleave(self.alpha.data, C // self.num_heads, dim=1)
                self.alpha.data = self.alpha.data.squeeze().unsqueeze(0).unsqueeze(0)
            x = self.alpha * x_attn_relu + (1 - self.alpha) * x_scalattn
        
        '''
        # XNorm Attention
        kv = k.transpose(-2, -1) @ v  # kv: (B, #heads, C // #heads, C // #heads)
        kv_normed = self.XNorm(kv, self.gamma_xnorm)  # kv_normed: (B, #heads, C // #heads, C // #heads)
        q_normed = self.XNorm(q, self.gamma_xnorm)  # q_normed: (B, #heads, HW + 1, C // #heads)
        x = (q_normed @ kv_normed).transpose(1, 2).reshape(B, N, C)  # x: (B, HW + 1, C)
        '''

        '''
        # Hydra Attention (cosine similarity)
        q/k/v: (B, HW+1, #heads), #heads = #channels
        k = k.squeeze(-1).transpose(-2, -1)
        q = q.squeeze(-1).transpose(-2, -1)
        v = v.squeeze(-1).transpose(-2, -1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv  # x: (B, HW + 1, C)
        '''

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # XNorm
    def XNorm(self, x, gamma_xnorm):
        norm_tensor = torch.norm(x, 2, -1, True)
        return x * gamma_xnorm / norm_tensor


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
        self.relu = F.relu
        # self.prelu = nn.PReLU(init=0.5)
        self.relu6 = F.relu6

        # self.beta = nn.Parameter(torch.ones(1, 257, 1), requires_grad=True)  # token-wise 

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))  # origin gelu

        # src2 = self.linear1(src)
        # src2 = self.beta * self.activation(src2) + (1 - self.beta) * src2  # search gelu
        # src2 = self.relu(src2)
        # self.beta.data = torch.clamp(self.beta.data, min=0., max=1.)

        # src2 = self.linear2(self.dropout1(src2))
        
        # src2 = (1 - self.beta) * self.activation(src2) + self.beta * src2  # relu after linear layers

        src = src + self.drop_path(self.dropout2(src2))
        return src


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


class TransformerClassifier(Module):
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
            x = self.fc(x)

        return x, features

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
