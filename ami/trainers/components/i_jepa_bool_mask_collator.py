import math
from multiprocessing import Value

import torch
from torch import Tensor
from torch.utils.data import default_collate

from ami.models.utils import size_2d, size_2d_to_int_tuple


class IJEPABoolMaskCollator:
    """I-JEPA collator function for providing boolean mask tensors.

    This collator creates boolean masks for both the context encoder and predictor.
    It's designed to work with the I-JEPA (Image Joint Embedding Predictive Architecture) model.

    The masks are boolean tensors where:
    - True values indicate patches to be masked (ignored)
    - False values indicate patches to be processed or predicted

    This differs from IJEPAMaskCollator which uses integer indices for masked patches.
    """

    def __init__(
        self,
        input_size: size_2d,
        patch_size: size_2d = 16,
        mask_scale: tuple[float, float] = (0.15, 0.50),
        n_masks: int = 4,
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_keep: int = 10,
    ) -> None:
        """Initialize the IJEPABoolMaskCollator.

        Args:
            input_size (size_2d): Size of the input image.
            patch_size (size_2d): Size of each patch.
            mask_scale (tuple[float, float]): Range of mask scale (min, max).
            n_masks (int): Number of mask candidates to generate.
            aspect_ratio (tuple[float, float]): Range of aspect ratios for masks.
            min_keep (int): Minimum number of patches to keep unmasked.
        """
        super().__init__()
        assert mask_scale[0] < mask_scale[1]
        assert mask_scale[0] > 0
        assert mask_scale[1] < 1

        input_size = size_2d_to_int_tuple(input_size)
        self.patch_size = size_2d_to_int_tuple(patch_size)

        assert input_size[0] % self.patch_size[0] == 0
        assert input_size[1] % self.patch_size[1] == 0

        self.n_patches_height = input_size[0] // self.patch_size[0]
        self.n_patches_width = input_size[1] // self.patch_size[1]
        assert min_keep <= self.n_patches_height * self.n_patches_width

        self.mask_scale = mask_scale
        self.n_masks = n_masks
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep  # minimum number of patches to keep unmasked
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self) -> int:
        """Increment and return the iteration counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_rectangle(
        self,
        generator: torch.Generator,
    ) -> tuple[int, int, int, int]:
        """Randomly sample a rectangular mask.

        Args:
            generator (torch.Generator): Generator for pseudo-random numbers.

        Returns:
            tuple[int, int, int, int]: Top, bottom, left, and right coordinates of the mask.
        """
        _scale_rand, _ratio_rand = torch.rand(2, generator=generator).tolist()
        # -- Sample mask scale
        min_s, max_s = self.mask_scale
        mask_scale = min_s + _scale_rand * (max_s - min_s)
        max_keep = int(self.n_patches_height * self.n_patches_width * mask_scale)

        # -- Sample mask aspect-ratio
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + _ratio_rand * (max_ar - min_ar)

        # -- Compute height and width of mask (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        h = min(self.n_patches_height, h)
        w = min(self.n_patches_width, w)

        # -- Compute mask coordinates
        top = int(torch.randint(high=self.n_patches_height - h + 1, size=(1,), generator=generator).item())
        left = int(torch.randint(high=self.n_patches_width - w + 1, size=(1,), generator=generator).item())
        bottom = top + h
        right = left + w
        return top, bottom, left, right

    def sample_masks(self, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """Sample boolean masks for the encoder and predictor.

        Args:
            generator (torch.Generator): Generator for pseudo-random numbers.

        Returns:
            tuple[Tensor, Tensor]: Boolean masks for encoder and predictor.
        """
        sampled_masks = []
        for _ in range(self.n_masks):
            mask = torch.zeros(self.n_patches_height, self.n_patches_width, dtype=torch.bool)
            top, bottom, left, right = self._sample_mask_rectangle(generator)
            mask[top:bottom, left:right] = True
            sampled_masks.append(mask.flatten())

        # Create encoder mask by combining all sampled masks
        encoder_mask = torch.stack(sampled_masks).sum(0).type(torch.bool)
        # Randomly select one mask as the predictor target
        predictor_target_mask = torch.logical_not(
            sampled_masks[int(torch.randint(high=len(sampled_masks), size=(1,), generator=generator).item())]
        )

        # Ensure minimum number of unmasked patches for encoder
        encoder_unmask_patch_count = encoder_mask.logical_not().sum()
        if encoder_unmask_patch_count < self.min_keep:
            encoder_mask[-self.min_keep :] = False
            predictor_target_mask[-self.min_keep :] = True

        # Sanity checks
        if encoder_mask.logical_not().sum() == 0:
            raise RuntimeError("Encoder mask is all True! No input for encoder!")
        if predictor_target_mask.logical_not().sum() == 0:
            raise RuntimeError("Predictor mask is all True! No prediction target for predictor!")

        return encoder_mask, predictor_target_mask

    def __call__(self, images: list[tuple[Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        """Collate input images and create boolean masks for context encoder
        and predictor.

        Args:
            images (list[tuple[Tensor]]): List of image tensors. Each image is shape [3, height, width].

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - collated_images: Collated images (shape: [batch_size, 3, height, width])
                - collated_masks_for_context_encoder: Boolean mask for context encoder (shape: [batch_size, n_patches])
                - collated_masks_for_predictor: Boolean mask for predictor (shape: [batch_size, n_patches])
        """
        collated_images: Tensor = default_collate(images)[0]

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        masks_for_encoder, masks_for_predictor = [], []
        for _ in range(len(images)):
            enc_mask, pred_mask = self.sample_masks(g)
            masks_for_encoder.append(enc_mask)
            masks_for_predictor.append(pred_mask)

        collated_masks_for_context_encoder = torch.stack(masks_for_encoder)
        collated_masks_for_predictor = torch.stack(masks_for_predictor)

        return (
            collated_images,
            collated_masks_for_context_encoder,
            collated_masks_for_predictor,
        )