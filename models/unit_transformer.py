import numpy as np
import torch
import torch.nn as nn


class CCF3DSinCosPositionEmbedding(nn.Module):
    def __init__(self, d_model=128, temperature=10000.0):
        super().__init__()

        self.d_model = d_model
        self.temperature = temperature
        self.num_axes = 3

        axis_dim = d_model // self.num_axes
        if axis_dim % 2 != 0:
            axis_dim -= 1

        self.axis_dim = axis_dim
        self.pe_dim = self.axis_dim * self.num_axes

        if self.pe_dim != d_model:
            self.proj = nn.Linear(self.pe_dim, d_model)
        else:
            self.proj = nn.Identity()

    def encode_one_axis(self, pos):
        device = pos.device
        half_dim = self.axis_dim // 2

        dim_t = torch.arange(
            half_dim,
            dtype=torch.float32,
            device=device,
        )
        dim_t = self.temperature ** (2 * dim_t / self.axis_dim)

        pos = pos.unsqueeze(-1) / dim_t
        pe = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)
        return pe

    def forward(self, ccf):
        # ccf: [B, N, 3] with AP/DV/LR coordinates.
        ap = ccf[:, :, 0]
        dv = ccf[:, :, 1]
        lr = ccf[:, :, 2]

        ap_pe = self.encode_one_axis(ap)
        dv_pe = self.encode_one_axis(dv)
        lr_pe = self.encode_one_axis(lr)

        pe = torch.cat([ap_pe, dv_pe, lr_pe], dim=-1)
        pe = self.proj(pe)
        return pe


class MaskedMeanPooling(nn.Module):
    def forward(self, x, mask):
        mask_float = mask.unsqueeze(-1).float()
        x = x * mask_float
        return x.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        score = self.score(x).squeeze(-1)          # [B, N]
        score = score.masked_fill(~mask, -1e9)     # padding excluded from softmax
        weight = torch.softmax(score, dim=1)       # [B, N]
        return torch.sum(x * weight.unsqueeze(-1), dim=1)  # [B, D]


class RegionWiseUnitSelfAttention(nn.Module):
    """
    Unit self-attention constrained within each target region.

    Input/output tokens stay [B, N, D]; only same-region real units attend each
    other before the global cross-region Transformer.
    """

    def __init__(
        self,
        d_model=128,
        num_regions=3,
        nhead=8,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        share_weights=False,
    ):
        super().__init__()
        self.num_regions = int(num_regions)
        self.share_weights = bool(share_weights)

        def make_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        if self.share_weights:
            self.shared_encoder = make_encoder()
            self.region_encoders = None
        else:
            self.shared_encoder = None
            self.region_encoders = nn.ModuleList([
                make_encoder() for _ in range(self.num_regions)
            ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens, region_ids, mask):
        """
        tokens:     [B, N, D]
        region_ids: [B, N]
        mask:       [B, N], True=real unit, False=padding
        """
        out = tokens.clone()

        for rid in range(self.num_regions):
            region_mask = mask & (region_ids == rid)
            if not bool(region_mask.any().item()):
                continue

            encoder = self.shared_encoder if self.share_weights else self.region_encoders[rid]
            encoded = encoder(tokens, src_key_padding_mask=~region_mask)
            out = torch.where(region_mask.unsqueeze(-1), encoded, out)

        out = self.out_norm(out)
        out = out * mask.float().unsqueeze(-1)
        return out


class UnitTransformerRetrievalEncoder(nn.Module):
    def __init__(
        self,
        proj_dim,
        d_model=128,
        mlp_hidden=256,
        num_regions=3,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.3,
        pooling="attn",
        use_region_self_attn=True,
        region_sa_layers=1,
        region_sa_nhead=4,
        region_sa_dim_feedforward=256,
        region_sa_share_weights=False,
    ):
        super().__init__()

        if int(region_sa_layers) < 0:
            raise ValueError(f"region_sa_layers cannot be less than 0, got: {region_sa_layers}")

        self.num_regions = num_regions
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.pooling = pooling
        self.region_sa_layers = int(region_sa_layers)
        self.use_region_self_attn = bool(use_region_self_attn) and self.region_sa_layers > 0
        self.region_sa_share_weights = bool(region_sa_share_weights)

        self.response_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ccf_position_embedding = CCF3DSinCosPositionEmbedding(
            d_model=d_model,
            temperature=10000.0,
        )

        self.region_embedding = nn.Embedding(num_regions, d_model)
        self.token_norm = nn.LayerNorm(d_model)

        if self.use_region_self_attn:
            self.region_self_attention = RegionWiseUnitSelfAttention(
                d_model=d_model,
                num_regions=num_regions,
                nhead=region_sa_nhead,
                num_layers=self.region_sa_layers,
                dim_feedforward=region_sa_dim_feedforward,
                dropout=dropout,
                share_weights=region_sa_share_weights,
            )
        else:
            self.region_self_attention = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        if pooling == "mean":
            self.pool = MaskedMeanPooling()
        elif pooling == "attn":
            self.pool = MaskedAttentionPooling(d_model)
        else:
            raise ValueError(f"Unknown pooling: {pooling}; expected mean or attn")

        self.projector = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, proj_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, response, ccf, region_ids, mask):
        """
        response:   [B, N]
        ccf:        [B, N, 3]
        region_ids: [B, N]
        mask:       [B, N], True=real unit, False=padding
        """
        response_emb = self.response_embedding(response.unsqueeze(-1))
        pos_emb = self.ccf_position_embedding(ccf)
        region_emb = self.region_embedding(region_ids.clamp(min=0, max=self.num_regions - 1))

        tokens = response_emb + pos_emb + region_emb
        # tokens = response_emb
        tokens = self.token_norm(tokens)
        tokens = tokens * mask.float().unsqueeze(-1)

        if self.use_region_self_attn:
            tokens = self.region_self_attention(tokens, region_ids, mask)

        tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        tokens = self.encoder_norm(tokens)
        tokens = tokens * mask.float().unsqueeze(-1)

        feat = self.pool(tokens, mask)       # [B, d_model]
        z = self.projector(feat)             # [B, proj_dim]
        return z
