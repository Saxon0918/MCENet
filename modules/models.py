import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

class MCENet(nn.Module):
    def __init__(self, hyp_params):
        super(MCENet, self).__init__()
        self.orig_d_m, self.orig_d_a, self.orig_d_f, self.orig_d_g = hyp_params.orig_d_m, hyp_params.orig_d_a, hyp_params.orig_d_f, hyp_params.orig_d_g
        self.d_m, self.d_a, self.d_f, self.d_g = 40, 40, 40, 40
        self.mri = hyp_params.mri
        self.av45 = hyp_params.av45
        self.fdg = hyp_params.fdg
        self.gene = hyp_params.gene
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.partial_mode = self.mri + self.av45 + self.fdg + self.gene
        combined_dim = self.d_m + self.d_a + self.d_f + self.d_g
        output_dim = hyp_params.output_dim

        # Linear Layer
        self.proj_m = nn.Linear(self.orig_d_m, self.d_m)  # 140->40
        self.proj_a = nn.Linear(self.orig_d_a, self.d_a)  # 140->40
        self.proj_f = nn.Linear(self.orig_d_f, self.d_f)  # 140->40
        self.proj_g = nn.Linear(self.orig_d_g, self.d_g)  # 140->40

        # Cross-Modal Information Enhancement Module
        if self.mri:
            self.trans_m_with_a = self.get_network(self_type='ma')  # 40->40
            self.trans_m_with_f = self.get_network(self_type='mf')  # 40->40
            self.trans_m_with_g = self.get_network(self_type='mg')  # 40->40
        if self.av45:
            self.trans_a_with_m = self.get_network(self_type='am')  # 40->40
            self.trans_a_with_f = self.get_network(self_type='af')  # 40->40
            self.trans_a_with_g = self.get_network(self_type='ag')  # 40->40
        if self.fdg:
            self.trans_f_with_m = self.get_network(self_type='fm')  # 40->40
            self.trans_f_with_a = self.get_network(self_type='fa')  # 40->40
            self.trans_f_with_g = self.get_network(self_type='fg')  # 40->40
        if self.gene:
            self.trans_g_with_m = self.get_network(self_type='gm')  # 40->40
            self.trans_g_with_a = self.get_network(self_type='ga')  # 40->40
            self.trans_g_with_f = self.get_network(self_type='gf')  # 40->40

        # Transformer
        self.trans_m_mem = self.get_network(self_type='m_mem', layers=3)  # 40->40
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)  # 40->40
        self.trans_f_mem = self.get_network(self_type='f_mem', layers=3)  # 40->40
        self.trans_g_mem = self.get_network(self_type='g_mem', layers=3)  # 40->40

        # Multimodality Fusion Layer
        self.proj1 = nn.Linear(combined_dim, combined_dim)  # 160->160
        self.proj2 = nn.Linear(combined_dim, combined_dim)  # 160->160
        self.out_layer = nn.Linear(combined_dim, output_dim)  # 160->1

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['m', 'ma', 'mf', 'mg']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        elif self_type in ['a', 'am', 'af', 'ag']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type in ['f', 'fm', 'fa', 'fg']:
            embed_dim, attn_dropout = self.d_f, self.attn_dropout
        elif self_type in ['g', 'gm', 'ga', 'gf']:
            embed_dim, attn_dropout = self.d_g, self.attn_dropout
        elif self_type == 'm_mem':
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'f_mem':
            embed_dim, attn_dropout = self.d_f, self.attn_dropout
        elif self_type == 'g_mem':
            embed_dim, attn_dropout = self.d_g, self.attn_dropout
        else:
            raise ValueError("Unknown type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  modal_num=self.partial_mode)

    def forward(self, x_m, x_a, x_f, x_g):
        # Linear Layer
        proj_x_m = x_m if self.orig_d_m == self.d_m else self.proj_m(x_m)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_f = x_f if self.orig_d_f == self.d_f else self.proj_f(x_f)
        proj_x_g = x_g if self.orig_d_g == self.d_g else self.proj_g(x_g)

        # Cross-Modal Information Enhancement Module and Transformer
        if self.mri:
            h_m_with_ma = self.trans_m_with_a(proj_x_m, proj_x_a, proj_x_a)
            h_m_with_maf = self.trans_m_with_f(h_m_with_ma, proj_x_f, proj_x_f)
            h_m_with_mafg = self.trans_m_with_g(h_m_with_maf, proj_x_g, proj_x_g)
            h_ms = h_m_with_mafg
            h_ms = self.trans_m_mem(h_ms)
            if type(h_ms) == tuple:
                h_ms = h_ms[0]
        if self.av45:
            h_a_with_am = self.trans_a_with_m(proj_x_a, proj_x_m, proj_x_m)
            h_a_with_amf = self.trans_a_with_f(h_a_with_am, proj_x_f, proj_x_f)
            h_a_with_amfg = self.trans_a_with_g(h_a_with_amf, proj_x_g, proj_x_g)
            h_as = h_a_with_amfg
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
        if self.fdg:
            h_f_with_fm = self.trans_f_with_m(proj_x_f, proj_x_m, proj_x_m)
            h_f_with_fma = self.trans_f_with_a(h_f_with_fm, proj_x_a, proj_x_a)
            h_f_with_fmag = self.trans_f_with_g(h_f_with_fma, proj_x_g, proj_x_g)
            h_fs = h_f_with_fmag
            h_fs = self.trans_f_mem(h_fs)
            if type(h_fs) == tuple:
                h_fs = h_fs[0]
        if self.gene:
            h_g_with_gm = self.trans_g_with_m(proj_x_g, proj_x_m, proj_x_m)
            h_g_with_gma = self.trans_g_with_a(h_g_with_gm, proj_x_a, proj_x_a)
            h_g_with_gmaf = self.trans_g_with_f(h_g_with_gma, proj_x_f, proj_x_f)
            h_gs = h_g_with_gmaf
            h_gs = self.trans_g_mem(h_gs)
            if type(h_gs) == tuple:
                h_gs = h_gs[0]

        # MLP
        if self.partial_mode == 4:
            last_hs = torch.cat([h_ms, h_as, h_fs, h_gs], dim=1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        return output, last_hs


class MCENet_My(nn.Module):
    def __init__(self, hyp_params):
        super(MCENet_My, self).__init__()
        self.orig_d_m, self.orig_d_t, self.orig_d_c = hyp_params.orig_d_m, hyp_params.orig_d_t, hyp_params.orig_d_c
        self.d_m, self.d_t, self.d_c = 40, 40, 40
        self.mri = hyp_params.mri
        self.ct = hyp_params.ct
        self.clinical = hyp_params.clinical
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.partial_mode = self.mri + self.ct + self.clinical
        combined_dim = self.d_m + self.d_t + self.d_c
        output_dim = hyp_params.output_dim

        # Linear Layer test
        self.proj_m1 = nn.Linear(self.orig_d_m, self.orig_d_m)  # 140->40
        self.proj_t1 = nn.Linear(self.orig_d_t, self.orig_d_t)  # 140->40
        self.proj_c1 = nn.Linear(self.orig_d_c, self.orig_d_c)  # 10->40

        # Linear Layer
        self.proj_m = nn.Linear(self.orig_d_m, self.d_m)  # 140->40
        self.proj_t = nn.Linear(self.orig_d_t, self.d_t)  # 140->40
        self.proj_c = nn.Linear(self.orig_d_c, self.d_c)  # 10->40

        # Cross-Modal Information Enhancement Module
        if self.mri:
            self.trans_m_with_t = self.get_network(self_type='mt')  # 40->40
            self.trans_m_with_c = self.get_network(self_type='mc')  # 40->40
        if self.ct:
            self.trans_t_with_m = self.get_network(self_type='tm')  # 40->40
            self.trans_t_with_c = self.get_network(self_type='tc')  # 40->40
        if self.clinical:
            self.trans_c_with_m = self.get_network(self_type='cm')  # 40->40
            self.trans_c_with_t = self.get_network(self_type='ct')  # 40->40

        # Transformer
        self.trans_m_mem = self.get_network(self_type='m_mem', layers=3)  # 40->40
        self.trans_t_mem = self.get_network(self_type='t_mem', layers=3)  # 40->40
        self.trans_c_mem = self.get_network(self_type='c_mem', layers=3)  # 40->40

        # Multimodality Fusion Layer
        self.proj1 = nn.Linear(combined_dim, combined_dim)  # 120->120
        self.proj2 = nn.Linear(combined_dim, combined_dim)  # 120->120
        self.out_layer = nn.Linear(combined_dim, output_dim)  # 120->1

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['m', 'mt', 'mc']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        elif self_type in ['t', 'tm', 'tc']:
            embed_dim, attn_dropout = self.d_t, self.attn_dropout
        elif self_type in ['c', 'cm', 'ct']:
            embed_dim, attn_dropout = self.d_c, self.attn_dropout
        elif self_type == 'm_mem':
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        elif self_type == 't_mem':
            embed_dim, attn_dropout = self.d_t, self.attn_dropout
        elif self_type == 'c_mem':
            embed_dim, attn_dropout = self.d_c, self.attn_dropout
        else:
            raise ValueError("Unknown type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  modal_num=self.partial_mode)

    def forward(self, x_m, x_t, x_c):
        proj_x_m = x_m if self.orig_d_m == self.d_m else self.proj_m1(x_m)
        proj_x_t = x_t if self.orig_d_t == self.d_t else self.proj_t1(x_t)
        proj_x_c = x_c if self.orig_d_c == self.d_c else self.proj_c1(x_c)

        # Linear Layer
        proj_x_m = x_m if self.orig_d_m == self.d_m else self.proj_m(proj_x_m)
        proj_x_t = x_t if self.orig_d_t == self.d_t else self.proj_t(proj_x_t)
        proj_x_c = x_c if self.orig_d_c == self.d_c else self.proj_c(proj_x_c)

        # Cross-Modal Information Enhancement Module and Transformer
        if self.mri:
            h_m_with_mt = self.trans_m_with_t(proj_x_m, proj_x_t, proj_x_t)
            h_m_with_mtc = self.trans_m_with_c(h_m_with_mt, proj_x_c, proj_x_c)
            h_ms = h_m_with_mtc
            h_ms = self.trans_m_mem(h_ms)
            if type(h_ms) == tuple:
                h_ms = h_ms[0]
        if self.ct:
            h_a_with_tm = self.trans_t_with_m(proj_x_t, proj_x_m, proj_x_m)
            h_a_with_tmc = self.trans_t_with_c(h_a_with_tm, proj_x_c, proj_x_c)
            h_ts = h_a_with_tmc
            h_ts = self.trans_t_mem(h_ts)
            if type(h_ts) == tuple:
                h_ts = h_ts[0]
        if self.clinical:
            h_f_with_cm = self.trans_c_with_m(proj_x_c, proj_x_m, proj_x_m)
            h_f_with_cmt = self.trans_c_with_t(h_f_with_cm, proj_x_t, proj_x_t)
            h_cs = h_f_with_cmt
            h_cs = self.trans_c_mem(h_cs)
            if type(h_cs) == tuple:
                h_cs = h_cs[0]
        # MLP
        if self.partial_mode == 3:
            last_hs = torch.cat([h_ms, h_ts, h_cs], dim=1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        return output, last_hs


class ConLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        similarity_matrix /= self.temperature

        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.T)  # positive
        negative_mask = ~positive_mask  # negtive
        pos_sim = similarity_matrix * positive_mask.float()
        neg_sim = similarity_matrix * negative_mask.float()

        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim).sum(dim=1, keepdim=True)
        loss = -torch.log(exp_pos / (exp_pos + exp_neg))
        loss = loss[positive_mask].mean()
        return loss