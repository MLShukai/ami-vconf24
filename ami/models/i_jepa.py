# Ref: https://github.com/facebookresearch/ijepa

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .components.patch_embedding import PatchEmbedding
from .components.positional_embeddings import get_2d_positional_embeddings
from .components.vision_transformer_layer import VisionTransformerLayer
from .model_wrapper import ModelWrapper
from .utils import size_2d, size_2d_to_int_tuple


def repeat_patches_along_with_batch_axis(
    x: torch.Tensor, batch_size: int, n_patch_selections_for_context_encoder: int
) -> torch.Tensor:
    """

    Args:
        x (torch.Tensor):
            Input patches.
            (shape is [batch_size * n_patch_selections_for_predictor, n_patches, dims])
        batch_size (int):
            batch size.
        n_patch_selections_for_context_encoder (int):
            num of patch selections for context encoder.

    Returns:
        torch.Tensor:
            Repeated patches along with first axis.
            (shape: [batch_size * n_patch_selections_for_context_encoder * n_patch_selections_for_predictor, n_patches, dims])
    """
    n_patch_selections_for_predictor = len(x) // batch_size
    out = []
    for i in range(n_patch_selections_for_predictor):
        out.append(
            torch.cat(
                [x[i * batch_size : (i + 1) * batch_size] for _ in range(n_patch_selections_for_context_encoder)], dim=0
            )
        )
    return torch.cat(out, dim=0)


def select_patches_by_indices(x: torch.Tensor, patch_selections: list[torch.Tensor]) -> torch.Tensor:
    """Extract patches from x using indices in patch_selections.

    Args:
        x (torch.Tensor):
            Input patches.
            (shape: [batch_size, n_patches, dims])
        patch_selections (list[torch.Tensor]):
            Indices to select patches.
            (len(patch_selections) is num of patch_selections, shape of all Tensors: [batch_size, n_patches_to_be_selected])

    Returns:
        torch.Tensor:
            patches Selected from input x.
            (shape: [batch_size*len(patch_selections), n_patches_to_be_selected, dims])
    """
    selected_x = []
    for patch_selection in patch_selections:
        indices = patch_selection.unsqueeze(-1).repeat(1, 1, x.size(-1))
        selected_x.append(torch.gather(x, dim=1, index=indices))
    return torch.cat(selected_x, dim=0)


class IJEPAEncoder(nn.Module):
    """Used as I-JEPA context_encoder and target_encoder."""

    def __init__(
        self,
        img_size: size_2d = 224,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Used as I-JEPA context_encoder and target_encoder.

        Args:
            img_size (size_2d):
                Input image size.
                Defaults to 224.
            patch_size (size_2d):
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
            qk_scale (float | None):
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
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        # define positional encodings
        img_size = size_2d_to_int_tuple(img_size)
        patch_size = size_2d_to_int_tuple(patch_size)
        img_height, img_width = img_size
        patch_height, patch_width = patch_size
        assert img_height % patch_height == 0
        assert img_width % patch_width == 0
        n_patches_hw = (img_height // patch_height), (img_width // patch_width)
        n_patches = n_patches_hw[0] * n_patches_hw[1]
        self.positional_encodings: torch.Tensor
        positional_encodings = get_2d_positional_embeddings(
            embed_dim,
            n_patches_hw,
        ).reshape(1, n_patches, embed_dim)
        self.register_buffer("positional_encodings", torch.from_numpy(positional_encodings).float())
        # define transformers
        dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule
        self.vit_layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # initialize
        self.init_std = init_std
        self.apply(partial(_init_weights, init_std=init_std))
        fix_init_weight(self.vit_layers)

    def forward(
        self,
        images: torch.Tensor,
        patch_selections_for_context_encoder: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode input images into latents.

        Args:
            images (torch.Tensor):
                Input images.
                (shape: [batch_size, 3, height, width])
            patch_selections_for_context_encoder (list[torch.Tensor] | None):
                Masks to select indices of images embedded as patches.
                (shape of each Tensor: [batch_size, n_patches])
                If None, all patches are selected and used to create latents.
                Default to None.

        Returns:
            torch.Tensor:
                if patch_selections_for_context_encoder is None:
                    shape: [batch_size, n_patches, embed_dim]
                else:
                    shape: [batch_size*len(patch_selections_for_context_encoder), n_patches, embed_dim]
        """
        # patchify input images
        x = self.patch_embed(images)
        # x: [batch_size, n_patches, embed_dim]

        # add positional embedding to x
        x = x + self.positional_encodings

        # Extract patches from x based on indices contained in patch_selections_for_context_encoder
        if patch_selections_for_context_encoder is not None:
            x = select_patches_by_indices(x, patch_selections_for_context_encoder)

        # Apply Vision Transformers
        for vit_layer in self.vit_layers:
            x = vit_layer(x)

        x = self.norm(x)

        return x


class IJEPAPredictor(nn.Module):
    """Used as I-JEPA predictor."""

    def __init__(
        self,
        n_patches: size_2d,
        context_encoder_embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Used as I-JEPA predictor.

        Args:
            n_patches (size_2d):
                Num of patches along with vertical and horizontal.
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
            qk_scale (float | None):
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
        # prepare tokens representing patches to be predicted
        self.token_for_prediction = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule
        # define positional encodings
        (n_patches_vertical, n_patches_horizontal) = size_2d_to_int_tuple(n_patches)
        self.positional_encodings: torch.Tensor
        positional_encodings = get_2d_positional_embeddings(
            predictor_embed_dim, grid_size=(n_patches_vertical, n_patches_horizontal)
        ).reshape(1, n_patches_vertical * n_patches_horizontal, predictor_embed_dim)
        self.register_buffer("positional_encodings", torch.from_numpy(positional_encodings).float())
        # define transformers
        self.vit_layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    embedding_dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
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
        torch.nn.init.trunc_normal_(self.token_for_prediction, std=self.init_std)
        self.apply(partial(_init_weights, init_std=init_std))
        fix_init_weight(self.vit_layers)

    def forward(
        self,
        latents: torch.Tensor,
        patch_selections_for_context_encoder: list[torch.Tensor],
        patch_selections_for_predictor: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict latents of patches at the position specified by
        patch_selections_for_predictor with a hint of input latents and
        positional information (patch_selections_for_context_encoder).

        Args:
            latents (torch.Tensor):
                Input latents from context_encoder.
                Since the first axis of output of Context Encoder (IJEPAEncoder) is batch_size*len(patch_selections_for_context_encoder),
                the first axis of latents is also batch_size*len(patch_selections_for_context_encoder) accordingly.
                (shape: [batch_size*len(patch_selections_for_context_encoder), n_patches_selected_in_context_encoder, context_encoder_embed_dim])
            patch_selections_for_context_encoder (list[torch.Tensor]):
                Masks which were used to create input latents above using context_encoder.
                (shape of each Tensor: [batch_size, n_patches_selected_in_context_encoder])
            patch_selections_for_predictor (list[torch.Tensor]):
                Masks corresponding to the position of the patches to be predict.
                (shape of each Tensor: [batch_size, n_patches_to_predict])

        Returns:
            torch.Tensor:
                prediction results.
                (shape: [batch_size*len(patch_selections_for_predictor), n_patches_to_predict, context_encoder_embed_dim])
        """

        # Since the first axis of output of Context Encoder (IJEPAEncoder)
        # is batch_size*len(patch_selections_for_context_encoder),
        # the first axis of latents is also batch_size*len(patch_selections_for_context_encoder) accordingly.
        assert len(latents) % len(patch_selections_for_context_encoder) == 0
        batch_size = len(latents) // len(patch_selections_for_context_encoder)

        # map from encoder-dim to pedictor-dim
        x = self.predictor_embed(latents)

        # add positional embedding correspond to context_encoder
        x_positional_encodings = self.positional_encodings.repeat(batch_size, 1, 1)
        x += select_patches_by_indices(x_positional_encodings, patch_selections_for_context_encoder)

        _, boundary, _ = x.shape

        # prepare positional embedding for patches to be predicted
        positional_encodings_for_prediction = self.positional_encodings.repeat(batch_size, 1, 1)
        positional_encodings_for_prediction = select_patches_by_indices(
            positional_encodings_for_prediction, patch_selections_for_predictor
        )
        positional_encodings_for_prediction = repeat_patches_along_with_batch_axis(
            positional_encodings_for_prediction,
            batch_size,
            n_patch_selections_for_context_encoder=len(patch_selections_for_context_encoder),
        )
        # prepare tokens for patch to be predicted
        tokens_for_prediction = self.token_for_prediction.repeat(
            positional_encodings_for_prediction.size(0), positional_encodings_for_prediction.size(1), 1
        )
        # add positional embedding
        tokens_for_prediction += positional_encodings_for_prediction
        # concat x with patches to be predicted
        x = x.repeat(len(patch_selections_for_predictor), 1, 1)
        x = torch.cat([x, tokens_for_prediction], dim=1)

        # Apply Vision Transformers
        for vit_layer in self.vit_layers:
            x = vit_layer(x)
        x = self.predictor_norm(x)

        # return predictions for patches correspond to indices in patch_selections_for_predictor
        x = x[:, boundary:]
        x = self.predictor_proj(x)

        return x


def fix_init_weight(vit_layers: nn.ModuleList) -> None:
    def rescale(param: torch.Tensor, layer_id: int) -> None:
        param.div_(math.sqrt(2.0 * layer_id))

    for layer_id, layer in enumerate(vit_layers, start=1):
        rescale(layer.attn.proj.weight.data, layer_id)
        rescale(layer.mlp.fc2.weight.data, layer_id)


def _init_weights(m: nn.Module, init_std: float) -> None:
    match m:
        case nn.Linear():
            torch.nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case nn.Conv2d():
            torch.nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def i_jepa_encoder_infer(wrapper: ModelWrapper[IJEPAEncoder], image: torch.Tensor) -> torch.Tensor:
    """Customizes the inference flow in the agent.

    Please specify to `ModelWrapper(inference_forward=<this>)`.
    Adding batch axis if image does not have it.

    Args:
        wrapper: ModelWrapper instance that wraps IJEPAEncoder.
        image: Input for IJEPAEncoder.
            shape (channel, height, width) or (batch, channel, height, width)

    Returns:
        torch.Tensor: Output of IJEPAEncoder.
            shape (patch, dim) or (batch, patch, dim)
    """
    device = wrapper.device
    no_batch = image.ndim == 3

    if no_batch:
        image = image.unsqueeze(0)  # batched

    image = image.to(device)

    out: torch.Tensor = wrapper(image)
    out = torch.nn.functional.layer_norm(out, (out.size(-1),))
    if no_batch:
        out = out.squeeze(0)

    return out
