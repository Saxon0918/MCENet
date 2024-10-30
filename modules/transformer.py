import torch
from torch import nn
import torch.nn.functional as F
from modules.multihead_attention import MultiheadAttention
import math


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = self.embed_scale * x_in
        x = F.dropout(x, p=self.dropout, training=self.training)
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        intermediates = [x]
        for layer in self.layers:
            # cross-modal or single-modal
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)
        if self.normalize:
            x = self.layer_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # 40->160
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)  # 160->40
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x  # residual连接
        x = self.maybe_layer_norm(0, x, before=True)  # 判断是before归一化还是after归一化
        if x_k is None and x_v is None:
            x = self.self_attn(query=x, key=x, value=x)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x = self.self_attn(query=x, key=x_k, value=x_v)  # 图3（a）整体结构，这是一个Muti-head的输出结果,X是embedding
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x  # 这是残差连接
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x  # 再记录一下残差，图3（b）上面黄色和紫色部分的残差
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))  # 40->160
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)  # 160->40
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x  # 再加上残差
        x = self.maybe_layer_norm(1, x, after=True)
        return x  # 这是一整个crossmodal transformer的结构，图3（b）整体结构

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after  # 必须确保要不之前归一化，要不之后归一化，只有一次
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m
