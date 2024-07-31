# Ref: https://github.com/facebookresearch/ijepa

import math
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    lower: float,
    upper: float,
) -> torch.Tensor:
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x: float) -> float:
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower_cdf = norm_cdf((lower - mean) / std)
        upper_cdf = norm_cdf((upper - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower_cdf - 1, 2 * upper_cdf - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=lower, max=upper)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, lower, upper)


def repeat_interleave_batch(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    N = len(x) // batch_size
    x = torch.cat(
        [torch.cat([x[i * batch_size : (i + 1) * batch_size] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )
    return x


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Extract patches from x using masks.

    Args:
        x (torch.Tensor):
            Input patches.
            (shape: [batch_size, n_patches, dims])
        masks (list[torch.Tensor]):
            Masks to select indices of patches.
            (len(masks) is num of masks, shape of all Tensors: [batch_size, n_patches_to_be_selected])

    Returns:
        torch.Tensor:
            patches Selected from input x.
            (shape: [batch_size, n_patches_to_be_selected, dims])
    """
    selected_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        selected_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(selected_x, dim=0)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> npt.NDArray[np.float64]:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    meshgrid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(meshgrid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob: float):
        super().__init__()
        assert drop_prob > 0.0
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = n_patches

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformerEncoder(nn.Module):
    """Used as I-JEPA context_encoder and target_encoder."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Used as I-JEPA context_encoder and target_encoder.

        Args:
            img_size (int):
                Input image size.
                Defaults to 224.
            patch_size (int):
                Pixel size per a patch.
                Defaults to 16.
            in_channels (int):
                Input images channels.
                Defaults to 3.
            embed_dim (int):
                Output dims per a patch.
                Defaults to 768.
            depth (int):
                Num of transformer layers.
                Defaults to 12.
            num_heads (int):
                Num of heads for transformer layers.
                Defaults to 12.
            mlp_ratio (float):
                Specify hidden dims for transformer layers.
                Defaults to 4.0.
            qkv_bias (bool):
                Whether to use bias in MLPs used to get qkv in attention module.
                Defaults to True.
            qk_scale (Optional[float]):
                The multiplier to be applied to the matrix product of q and v in attention module.
                Defaults to None.
            drop_rate (float):
                Ratio of Dropout to be performed within last MLP in each transformer layers.
                Defaults to 0.0.
            attn_drop_rate (float):
                Ratio of Dropout to be performed within MLP for the matrix product of q and k in attention module.
                Defaults to 0.0.
            drop_path_rate (float):
                Maximum value of stochastic depth decay rule.
                Defaults to 0.0.
            init_std (float):
                Std for initializing layers.
                Defaults to 0.02.
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # define input layer to convert input image into patches.
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        # define positional embeddings
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.n_patches**0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # define transformers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
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
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # initialize
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
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

    def forward(
        self, images: torch.Tensor, masks_for_context_encoder: Optional[list[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Encode input images into latents.

        Args:
            images (torch.Tensor):
                Input images.
                (shape: [batch_size, 3, img_size, img_size])
            masks_for_context_encoder (list[torch.Tensor] | None):
                Masks to select indices of images embedded as patches.
                (shape of each Tensor: [batch_size, n_patches])
                If None, all patches are selected and used to create latents.
                Default to None.

        Returns:
            torch.Tensor:
                if masks_for_context_encoder is None:
                    shape: [batch_size, n_patches, embed_dim]
                else:
                    shape: [batch_size*len(masks_for_context_encoder), n_patches, embed_dim]
        """
        # patchify input images
        x = self.patch_embed(images)
        # x: [batch_size, n_patches, embed_dim]

        # add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # Extract patches from x based on indices contained in masks_for_context_encoder
        if masks_for_context_encoder is not None:
            x = apply_masks(x, masks_for_context_encoder)

        # Apply Transformers
        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
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
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


class VisionTransformerPredictor(nn.Module):
    """Used as I-JEPA predictor."""

    def __init__(
        self,
        n_patches: int,
        context_encoder_embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Used as I-JEPA predictor.

        Args:
            n_patches (int):
                Total num of patches.
            context_encoder_embed_dim (int):
                Embedding dims of input latents.
                Defaults to 768.
            predictor_embed_dim (int):
                Convert the input dimension to this dimension for prediction.
                Defaults to 384.
            depth (int):
                Num of transformer layers.
                Defaults to 6.
            num_heads (int):
                Num of heads for transformer layers.
                Defaults to 12.
            mlp_ratio (float):
                Specify hidden dims for transformer layers.
                Defaults to 4.0.
            qkv_bias (bool):
                Whether to use bias in MLPs used to get qkv in attention module.
                Defaults to True.
            qk_scale (Optional[float]):
                The multiplier to be applied to the matrix product of q and v in attention module.
                Defaults to None.
            drop_rate (float):
                Ratio of Dropout to be performed within last MLP in each transformer layers.
                Defaults to 0.0.
            attn_drop_rate (float):
                Ratio of Dropout to be performed within MLP for the matrix product of q and k in attention module.
                Defaults to 0.0.
            drop_path_rate (float):
                Maximum value of stochastic depth decay rule.
                Defaults to 0.0.
            init_std (float):
                Std for initializing layers.
                Defaults to 0.02.
        """

        super().__init__()
        self.predictor_embed = nn.Linear(context_encoder_embed_dim, predictor_embed_dim, bias=True)
        # prepare mask tokens representing patches to be predicted
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # prepare positional embeddings
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, n_patches, predictor_embed_dim), requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1], int(n_patches**0.5), cls_token=False
        )
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # define transformers
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, context_encoder_embed_dim, bias=True)
        # initialize
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
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

    def forward(
        self,
        latents: torch.Tensor,
        masks_for_context_encoder: list[torch.Tensor],
        masks_for_predictor: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict latents of patches at the position specified by
        masks_for_predictor with a hint of input latents and positional
        information (masks_for_context_encoder).

        Args:
            latents (torch.Tensor):
                Input latents from context_encoder.
                (shape: [batch_size, n_patches_selected_in_context_encoder, context_encoder_embed_dim])
            masks_for_context_encoder (list[torch.Tensor]):
                Masks used to create input latents above using context_encoder.
                (shape of each Tensor: [batch_size, n_patches_selected_in_context_encoder])
            masks_for_predictor (list[torch.Tensor]):
                Masks corresponding to the position of the patches to be predict.
                (shape of each Tensor: [batch_size, n_patches_to_predict])

        Returns:
            torch.Tensor:
                prediction results.
                (shape: [batch_size*len(masks_for_predictor), n_patches_to_predict, context_encoder_embed_dim])
        """
        assert (masks_for_predictor is not None) and (
            masks_for_context_encoder is not None
        ), "Cannot run predictor without mask indices"

        batch_size = len(latents) // len(masks_for_context_encoder)

        # map from encoder-dim to pedictor-dim
        x = self.predictor_embed(latents)

        # add positional embedding correspond to context_encoder
        x_pos_embed = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        x += apply_masks(x_pos_embed, masks_for_context_encoder)

        _, boundary, _ = x.shape

        # prepare positional embedding for patches to be predicted
        pos_embs = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_for_predictor)
        pos_embs = repeat_interleave_batch(pos_embs, batch_size, repeat=len(masks_for_context_encoder))
        # prepare mask tokens for patch to be predicted
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # add positional embedding
        pred_tokens += pos_embs
        # concat x with patches to be predicted
        x = x.repeat(len(masks_for_predictor), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Apply Transformers
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # return predictions for mask tokens
        x = x[:, boundary:]
        x = self.predictor_proj(x)

        return x
