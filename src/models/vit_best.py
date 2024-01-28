import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)



class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        decoder_embed_dim=256,
        decoder_num_heads=2,
        decoder_depth=2,
        **kwargs
    ):
        super().__init__()

        # ---------------------------------------------------------------------- #
        # Encoder settings
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Set up the stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Set up the encoder blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ----------------------------------------------------------------------

        # ---------------------------------------------------------------------- #
        # Patch settings
        self.patch_embed = PatchEmbed(img_size=img_size[0],
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # ----------------------------------------------------------------------

        # ---------------------------------------------------------------------- #
        # Position settings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # ----------------------------------------------------------------------

        # ---------------------------------------------------------------------- #
        # Mask token settings
        self.mask_pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        # ----------------------------------------------------------------------

        # ---------------------------------------------------------------------- #
        # Decoder settings (a light weight decoder just for position prediction)
        # Require additional parameters:
        # - decoder_emebed_dim
        # - decoder_num_heads
        # - decoder_depth
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=decoder_num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_patches, bias=True)
        # ----------------------------------------------------------------------

        # ---------------------------------------------------------------------- #
        # Weight Initialiazation
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
        # ----------------------------------------------------------------------

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        # $$$$ Also initialize the mask_pos_token
        torch.nn.init.normal_(self.mask_pos_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def apply_pos_drop_mask(self, x, pos_embed, mask_pos_token, mask, pos_drop_ratio):
        B, N, D = x.shape
        device = x.device

        # Determine the number of positions to drop in the masked area
        num_pos_to_drop = int(mask.size(1) * pos_drop_ratio)

        # Shuffle mask along the last dimension
        random_tensor = torch.rand(B, mask.size(1), device=device)
        shuffled_indices = random_tensor.argsort(dim=1)
        shuffled_mask = mask.gather(1, shuffled_indices)

        # Split the mask into two: one for keeping pos_embed, one for mask_pos_token
        mask_no_pos = shuffled_mask[:, :num_pos_to_drop]
        mask_keep_pos = shuffled_mask[:, num_pos_to_drop:]

        # Apply the masks to x
        x_no_pos = apply_masks(x, [mask_no_pos])
        x_keep_pos = apply_masks(x, [mask_keep_pos])

        # Apply pos_embed and mask_pos_token accordingly
        mask_pos_tokens = mask_pos_token.repeat(B, num_pos_to_drop, 1).to(device)
        x_no_pos = x_no_pos + mask_pos_tokens

        pos_embed = pos_embed.repeat(B, 1, 1).to(device)
        pos_embed_masked = apply_masks(pos_embed, [mask_keep_pos])
        x_keep_pos = x_keep_pos + pos_embed_masked

        # Concatenate the results and shuffle again to restore the original order
        x = torch.cat([x_no_pos, x_keep_pos], dim=1)
        restored_indices = torch.argsort(shuffled_indices, dim=1)
        x = x.gather(1, restored_indices.unsqueeze(-1).expand(-1, -1, D))

        return x

    def apply_pos_drop_mask(self, x, pos_embed, mask_pos_token, mask, pos_drop_ratio):
        B, _, D = x.shape  # Original shape of x
        device = x.device

        # Determine the number of positions to drop in the masked area
        N_m = mask.size(1)  # Number of patches to keep after the mask is applied
        num_pos_to_drop = int(N_m * pos_drop_ratio)

        # Shuffle mask along the last dimension
        random_tensor = torch.rand(B, N_m, device=device)
        shuffled_indices = random_tensor.argsort(dim=1)
        shuffled_mask = mask.gather(1, shuffled_indices)

        # Split the mask into two: one for keeping pos_embed, one for mask_pos_token
        mask_no_pos = shuffled_mask[:, :num_pos_to_drop]
        mask_keep_pos = shuffled_mask[:, num_pos_to_drop:]

        # Apply the masks to x
        x_no_pos = apply_masks(x, [mask_no_pos])
        x_keep_pos = apply_masks(x, [mask_keep_pos])

        # Apply pos_embed and mask_pos_token accordingly
        mask_pos_tokens = mask_pos_token.repeat(B, num_pos_to_drop, 1).to(device)
        x_no_pos = x_no_pos + mask_pos_tokens

        pos_embed = pos_embed.repeat(B, 1, 1).to(device)
        pos_embed_masked = apply_masks(pos_embed, [mask_keep_pos])
        x_keep_pos = x_keep_pos + pos_embed_masked

        # Concatenate the results and shuffle again to restore the original order
        x = torch.cat([x_no_pos, x_keep_pos], dim=1)
        restored_indices = torch.argsort(shuffled_indices, dim=1)
        x = x.gather(1, restored_indices.unsqueeze(-1).expand(-1, -1, D))

        # Create a boolean mask in the shuffled order
        shuffled_pos_drop_mask = torch.zeros(B, N_m, dtype=torch.bool, device=device)
        shuffled_pos_drop_mask[:, :num_pos_to_drop] = True  # Mark the first num_pos_to_drop as True

        # Restore the order of the boolean mask to match x_restored
        pos_bool = shuffled_pos_drop_mask.gather(1, restored_indices)

        # The pos_drop_bool is used to apply on x to get the ones
        # whose positional embeddings are dropped
        # to apply it, you should you use it like
        # x_ = x[pos_drop_bool.unsqueeze(-1).expand(-1, -1, D)].reshape(B, -1, D)
        # Differently, mask_no_pos contains the original indices (no. of the patch)
        # and will be used as labels
        pos_labels = torch.sort(mask_no_pos.detach(), dim=1).values

        return x, pos_bool, pos_labels


    def forward_decoder(self, x):

        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # from decoder_embed_dim to num_patches

        return x

    def forward(self, x, masks=None, pos_drop_ratio=0, use_decoder=False):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)

        # When we do not drop the positional embeddings:
        if not pos_drop_ratio:
            x += pos_embed

            if masks is not None:
                x = apply_masks(x, masks)

        else:
            assert len(masks) == 1, 'Only one mask is needed for the context.'
            x, pos_bool, pos_labels = self.apply_pos_drop_mask(
                x, pos_embed, self.mask_pos_token, masks[0], pos_drop_ratio)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        if use_decoder:
            assert pos_drop_ratio, 'The function is only tested when pos are dropped.'
            logits = self.forward_decoder(x)
            return x, logits, pos_bool, pos_labels

        else:
            return x  # The classical IJEPA

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}
