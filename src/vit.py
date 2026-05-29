import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        # CIFAR 图片是 32x32；patch_size=4 时，会得到 8x8=64 个 patch。
        self.num_patches = (image_size // patch_size) ** 2
        # 用卷积实现切 patch：kernel_size 和 stride 都等于 patch_size。
        # 输出形状从 [B, 3, 32, 32] 变成 [B, embed_dim, 8, 8]。
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        # flatten 后把每个 patch 看成一个 token：
        # [B, embed_dim, 8, 8] -> [B, embed_dim, 64] -> [B, 64, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # cls_token 是可学习的分类 token，Transformer 输出后用它代表整张图片。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码让模型知道每个 patch 在图片中的空间位置。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # PyTorch 内置 TransformerEncoderLayer 已包含多头自注意力和 MLP。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # ViT 常用截断正态分布初始化可学习 token、位置编码和线性层权重。
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.size(0)
        # 先把图片切成 patch token。
        x = self.patch_embed(x)

        # 给每张图片都复制一个 cls_token，并拼到 patch token 的最前面。
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 加位置编码后，token 才包含顺序/空间位置信息。
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer 编码所有 token，再取 cls_token 的输出做分类。
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


def build_vit_tiny(num_classes=10):
    # 这是一个适合 CIFAR 的轻量 ViT：参数量小，训练速度相对友好。
    return ViT(
        image_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    )


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
