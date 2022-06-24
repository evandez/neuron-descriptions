"""Models for reranking MILAN descriptions.

Using a text-image constrative model like CLIP can improve MILAN's outputs.
This is a bit tricky, because (a) we are not just dealing with single images
and text, but *masked* images and text, and (b) the text must match a *set*
of masked images.

To make this work, we will sample many candidate descriptions from MILAN,
and then ask CLIP to rerank them according to how much each description
separately aligns with each top-activating image region and the image
as a whole.

The full reranking procedure:
    1. Sample N descriptions from the base MILAN model.
    2. Feed each image separately to CLIP.
        2.1 When CLIP computes the CLS token, downsample the image's activation
            mask and apply it to the attention weights for that token.
        2.2 Proceed, obtaining: p_masked_i = CLIP(image_i, mask_i, texts)
    3. Repeat above, without step (2.1), to get p_i = CLIP(image_i, texts)
    3. Rerank descriptions according to (1 - z) * p_masked_i + z * p_i
       for hyperparameter z in [0, 1]

"""
import math
from typing import Any, NamedTuple, Optional, Sequence, Tuple

from src.deps.netdissect import nethook, renormalize
from src.utils.typing import StrSequence

import clip
import torch
from torch import nn
from torch.nn import functional


class CLIPHookableMultiheadAttention(nn.Module):
    """Hookable version of PyTorch MultiheadAttention module.

    Only designed to work with CLIP. Ignores lots and lots of args!
    """

    def __init__(self, module: nn.MultiheadAttention):
        """Wrap the multihead attention module."""
        super().__init__()

        self.num_heads = module.num_heads

        # Copy raw qkv matrix to an true linear module.
        self.qkv = nn.Linear(module.in_proj_weight.shape[1],
                             module.in_proj_weight.shape[0])
        self.qkv.weight.data[:] = module.in_proj_weight.data
        self.qkv.bias.data[:] = module.in_proj_bias.data

        # Make softmax a true module as well.
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = module.out_proj

    def forward(self, hiddens: torch.Tensor, *_args: Any,
                **_kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attenutate over hidden representations.

        Args:
            hiddens (torch.Tensor): Hidden representations of shape
                (num_tokens, batch_size, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Hidden representations after
                self-attention layer, with same shape as input, and attention
                weights.

        """
        num_tokens, batch_size, hidden_size = hiddens.shape
        head_size = hidden_size // self.num_heads

        queries, keys, values = self.qkv(hiddens).chunk(3, dim=-1)
        queries = queries\
            .contiguous()\
            .view(num_tokens, batch_size * self.num_heads, head_size)\
            .transpose(0, 1)
        keys = keys\
            .contiguous()\
            .view(keys.shape[0], batch_size * self.num_heads, head_size)\
            .transpose(0, 1)
        values = values\
            .contiguous()\
            .view(values.shape[0], batch_size * self.num_heads, head_size)\
            .transpose(0, 1)

        queries = queries / math.sqrt(queries.shape[-1])
        attn = torch.bmm(queries, keys.transpose(-2, -1))
        attn = self.softmax(attn)

        output = torch.bmm(attn, values)
        output = output\
            .transpose(0, 1)\
            .contiguous()\
            .view(num_tokens, batch_size, hidden_size)
        output = self.out_proj(output)

        return output, attn


class CLIPWithMasks(nn.Module):
    """CLIP model that can use masks."""

    def __init__(self,
                 mask_layers: Optional[Sequence[int]] = None,
                 source_mean: Optional[Sequence[float]] = None,
                 source_std: Optional[Sequence[float]] = None,
                 **kwargs: Any):
        """Initialize the masked CLIP model.

        Args:
            mask_layers (Optional[Sequence[int]], optional): Layers to mask
                CLS token at in visual encoder. Defaults to all layers.
            source_mean (Optional[Sequence[float]]): Incoming normalization
                mean, to be corrected for. Defaults to no normalization.
            source_std (Optional[Sequence[float]]): Incoming normalization
                std, to be corrected for. Defaults to no normalization.

        """
        super().__init__()

        if (source_mean is None) != (source_std is None):
            raise ValueError('set neither or both of source_mean/source_std')

        # Load model and make attention hookable using an ugly wrapper.
        self.model, preprocess = clip.load(**kwargs)
        self.model.eval()

        for block in self.model.visual.transformer.resblocks:
            block.attn = CLIPHookableMultiheadAttention(block.attn)

        # Determine normalization scheme.
        if source_mean is None or source_std is None:
            source_mean, source_std = renormalize.OFFSET_SCALE['pt']
        self.renormalizer = renormalize.Renormalizer(
            source_mean,
            source_std,
            preprocess.transforms[-1].mean,
            preprocess.transforms[-1].std,
        )

        # Determine layers to mask.
        if mask_layers is None:
            self.mask_layers: Sequence[int] = tuple(
                range(len(self.model.visual.transformer.resblocks)))
        else:
            self.mask_layers = mask_layers

    def forward(self,
                images: torch.Tensor,
                texts: StrSequence,
                masks: torch.Tensor = None,
                resize: bool = True,
                renormalize: bool = True) -> torch.Tensor:
        """Compute similarity between the given images and texts.

        Args:
            images (torch.Tensor): The images. Must have shape
                (batch_size, channels, height, width).
            texts (StrSequence): The candidate texts.
            masks (torch.Tensor, optional): Optional masks corresponding
                to the images, capturing the regions to match text to.
                Should have shape (batch_size, 1, height, width).
                Defaults to None.
            resize (bool, optional): Resize images to appropriate resolution.
                Defaults to True.
            renormalize (bool, optional): Renormalize the images. Defaults
                to True.

        Returns:
            torch.Tensor: Scores with shape (batch_size, len(texts)).
                Normalized along last dimension so each element is a
                probability distribution over the candidate texts.

        """
        image_inputs = images
        if resize:
            image_inputs = functional.interpolate(image_inputs,
                                                  size=(self.input_resolution,
                                                        self.input_resolution),
                                                  mode='bicubic',
                                                  align_corners=False)
        if renormalize:
            image_inputs = self.renormalizer(images)

        text_inputs = torch\
            .cat([clip.tokenize(text) for text in texts])\
            .to(images.device)

        with nethook.InstrumentedModel(self.model) as instrumented:
            if masks is not None:
                num_patches_xy = self.num_patches_xy
                masks = functional\
                    .interpolate(masks,
                                 size=(num_patches_xy, num_patches_xy),
                                 mode='bilinear',
                                 align_corners=False)\
                    .view(len(masks), 1, self.num_patches)

                def rule(attentions: torch.Tensor) -> torch.Tensor:
                    assert masks is not None

                    attentions_by_head = attentions.view(
                        len(masks), -1, *attentions.shape[-2:])

                    attentions_masked = attentions_by_head[:, :, 0, 1:] * masks
                    attentions_masked = attentions_masked.view(
                        attentions.shape[0], -1)

                    attentions[:, 0, 1:] = attentions_masked
                    return attentions

                for layer in self.mask_layers:
                    instrumented.edit_layer(
                        f'visual.transformer.resblocks.{layer}.attn.softmax',
                        rule=rule)

            images_encoded = self.model.encode_image(image_inputs)
            images_encoded /= images_encoded.norm(dim=-1, keepdim=True)

            texts_encoded = self.model.encode_text(text_inputs)
            texts_encoded /= texts_encoded.norm(dim=-1, keepdim=True)

            similarities = images_encoded[:, None]\
                .mul(texts_encoded[None])\
                .sum(dim=-1)
            return similarities

    @property
    def num_patches(self) -> int:
        """Return number of patches used by CLIP ViT."""
        num_pixels = self.model.visual.input_resolution**2
        num_pixels_in_patch = self.model.visual.conv1.kernel_size[0]**2
        assert num_pixels % num_pixels_in_patch == 0, 'bad patch size?'
        return num_pixels // num_pixels_in_patch

    @property
    def num_patches_xy(self) -> int:
        """Return number of patches in each dimension of patch grid."""
        num_patches = self.num_patches
        size = math.isqrt(num_patches)
        assert size**2 == num_patches, 'non-square number of patches'
        return size

    @property
    def input_resolution(self) -> int:
        """Return input resolution for CLIP model."""
        return self.model.visual.input_resolution


class RerankerOutput(NamedTuple):
    """Output of a reranking algorithm."""

    texts: Sequence[StrSequence]
    orders: Sequence[Sequence[int]]
    scores: Sequence[Sequence[float]]


class CLIPWithMasksReranker(nn.Module):
    """Rerank sampled captions using CLIP."""

    def __init__(self, clip_with_masks: CLIPWithMasks, lam: float = .5):
        """Initialize the reranker.

        Args:
            clip_with_masks (CLIPWithMasks): The CLIP
            lam (float, optional): Default lambda value.
                See definition of forward fn.

        """
        super().__init__()
        self.clip_with_masks = clip_with_masks
        self.lam = lam

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        texts: Sequence[StrSequence],
        lam: Optional[float] = None,
    ) -> RerankerOutput:
        """Use MILAN to sample captions, then rerank them using CLIP scores.

        A smaller batch size is ideal, as this function is expensive!
        The kwargs are forwarded to the decoder.

        Args:
            images (torch.Tensor): The top-activating images. Should have shape
                (batch_size, k, channels, height, width).
            masks (torch.Tensor): The activation masks. Should have shape
                (batch_size, k, 1, height, width).
            texts (Sequence[StrSequence]): The candidate texts. Should be
                batch_size lists of arbitrary length.
            lam (float, optional): Trade off between masked and unmasked CLIP
                similarity scores. Defaults to .5 (equal weight).

        Returns:
            RerankerOutput: The reranked texts and corresponding indices.

        """
        if len(images) != len(masks):
            raise ValueError('images and masks batch sizes do not align: '
                             f'{len(images)} vs. {len(masks)}')
        if len(images) != len(texts):
            raise ValueError('images and texts batch sizes do not align: '
                             f'{len(images)} vs. {len(texts)}')

        if lam is None:
            lam = self.lam

        rerankeds, orders, scores = [], [], []
        for b_images, b_masks, b_texts in zip(images, masks, texts):
            sim_masked = self.clip_with_masks(b_images, b_texts, masks=b_masks)
            sim_masked = sim_masked.sum(dim=0)

            sim_unmasked = self.clip_with_masks(b_images, b_texts)
            sim_unmasked = sim_unmasked.sum(dim=0)

            sim = (1. - lam) * sim_masked + lam * sim_unmasked

            scoring, indices = sim.sort(descending=True)

            reranked = [b_texts[index] for index in indices.tolist()]
            rerankeds.append(tuple(reranked))
            orders.append(tuple(indices.tolist()))
            scores.append(tuple(scoring.tolist()))

        return RerankerOutput(tuple(rerankeds), tuple(orders), tuple(scores))


def reranker(lam: float = 1., **kwargs: Any) -> CLIPWithMasksReranker:
    """Create a new CLIPWithMasksReranker.

    The **kwargs are forwarded to CLIPWithMasks.
    """
    clip_with_masks = CLIPWithMasks(**kwargs)
    return CLIPWithMasksReranker(clip_with_masks, lam=lam)
