import math

import torch
from einops import rearrange
from torch import nn

from im2flow2act.common.vision_transformer import Block, trunc_normal_
from im2flow2act.diffusion_model.attention import BasicTransformerBlock


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device=device)
        batch_indices = rearrange(batch_indices, "... -> ... 1")
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = (
            torch.randn(b, n, device=device).topk(num_patches_keep, dim=-1).indices
        )

        return x[batch_indices, patch_indices_keep]


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the input for as many time
    steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000, frequency=10000.0) -> None:
        """
        Positional Encoding with maximum length
        :param size: embeddings dimension size
        :param max_len: maximum sequence length
        """
        if size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={size})"
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, size, 2, dtype=torch.float)
                * -(math.log(frequency) / size)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, size)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """
        Embed inputs.
        :param emb: (Tensor) Sequence of word embeddings vectors
            shape (batch_size,seq_len, dim)
        :return: positionally encoded word embeddings
        """
        # get position encodings
        return self.pe[:, : emb.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        dim_out=None,
        depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        use_norm_layer_at_the_end=True,
        learn_positional_encoding=False,
        context_length=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_out = dim_out
        self.use_norm_layer_at_the_end = use_norm_layer_at_the_end
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.learn_positional_encoding = learn_positional_encoding
        if self.learn_positional_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + context_length, embed_dim))
        else:
            self.pos_embed = PositionalEncoding(size=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if self.use_norm_layer_at_the_end:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

        trunc_normal_(self.cls_token, std=0.02)
        if self.dim_out is not None:
            self.proj_out = nn.Linear(embed_dim, self.dim_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, n, _ = x.shape
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        if self.learn_positional_encoding:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed(x)
        return self.pos_drop(x)

    def forward(self, x, return_cls_token=True):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.dim_out is not None:
            x = self.proj_out(x)

        if return_cls_token:
            return x[:, 0], x[:, 1:]
        else:
            return x[:, 1:]


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth=8,
        num_heads=12,
        d_head=64,
        dropout=0.0,
        class_token=True,
        context_dim=None,
        norm_layer=nn.LayerNorm,
        learn_positional_encoding=False,
        input_length=None,
        context_length=None,
        disable_self_attn=False,
        enable_position_encoding=False,
        dim_out=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_out = dim_out
        self.class_token = class_token
        self.enable_position_encoding = enable_position_encoding
        self.learn_positional_encoding = learn_positional_encoding
        self.prefix_len = 1 if self.class_token else 0
        if self.enable_position_encoding:
            if self.learn_positional_encoding:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.prefix_len + input_length, embed_dim)
                )
                self.context_pos_embed = nn.Parameter(
                    torch.zeros(1, context_length, context_dim)
                )
            else:
                self.pos_embed = PositionalEncoding(size=embed_dim)
                self.context_pos_embed = PositionalEncoding(size=context_dim)

        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=embed_dim,
                    n_heads=num_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    gated_ff=True,
                    checkpoint=False,
                    disable_self_attn=disable_self_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        if self.dim_out is not None:
            self.proj_out = nn.Linear(embed_dim, self.dim_out)
        if self.class_token:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, context):
        # discriptor: BxNxD; condition: BxNxD
        B, n, _ = x.shape
        if self.class_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.enable_position_encoding:
            if self.learn_positional_encoding:
                x = x + self.pos_embed
                context = context + self.context_pos_embed
            else:
                x = x + self.pos_embed(x)
                context = context + self.context_pos_embed(context)
        for blk in self.blocks:
            x = blk(x, context=context)
        x = self.norm(x)
        if self.dim_out is not None:
            x = self.proj_out(x)
        return x
