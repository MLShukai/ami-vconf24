import pytest
import torch
import random

from typing import Optional

from ami.models.i_jepa import (
    VisionTransformerEncoder,
    VisionTransformerPredictor,
)


class TestVisionTransformer:
    
    def _make_masks(self, n_mask: int, batch_size: int, n_patch_max: int) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []
        n_patch = random.randrange(n_patch_max)
        for _ in range(n_mask):
            m = []
            for _ in range(batch_size):
                m_indices, _ = torch.randperm(n_patch_max)[:n_patch].sort()
                m.append(m_indices)
            masks.append(torch.stack(m, dim=0))
        return masks

    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize(
        ["embed_dim", "depth", "num_heads", "mlp_ratio"],
        [
            [192, 12, 3, 4],  # tiny
            [384, 12, 6, 4],  # small
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_masks_for_encoder", [None, 1, 4])
    def test_vision_transformer_encoder(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        n_masks_for_encoder: Optional[int],
    ):
        assert image_size % patch_size == 0
        # define encoder made of ViT
        encoder = VisionTransformerEncoder(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # define sample inputs
        images = torch.randn([batch_size, 3, image_size, image_size])
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patch_max = n_patch_vertical*n_patch_horizontal
        # make masks for encoder
        masks_for_context_encoder = (
            self._make_masks(n_mask=n_masks_for_encoder, batch_size=batch_size, n_patch_max=n_patch_max) 
            if isinstance(n_masks_for_encoder, int) 
            else None
        )
        # get latents
        latent = encoder(images, masks_for_context_encoder)
        # check size of output latent
        expected_batch_size = (
            batch_size*n_masks_for_encoder
            if isinstance(n_masks_for_encoder, int) 
            else batch_size
        )
        assert latent.size(0) == expected_batch_size, "batch_size mismatch"
        expected_n_patch = (
            masks_for_context_encoder[0].size(-1)
            if isinstance(n_masks_for_encoder, int) 
            else n_patch_vertical * n_patch_horizontal
        )
        assert latent.size(1) == expected_n_patch, "num of patch mismatch"
        assert latent.size(2) == embed_dim, "embed_dim mismatch"
