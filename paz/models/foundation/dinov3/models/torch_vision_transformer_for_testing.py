import torch
import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
from torch import Tensor, nn
from paz.models.foundation.dinov3.utils import named_apply
from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_LayerScale,
    PT_Mlp,
    PT_PatchEmbed,
    PT_RMSNorm,
    PT_RopePositionEmbedding,
    PT_SelfAttentionBlock,
    PT_SwiGLUFFN,
)


logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": PT_Mlp,
    "swiglu": PT_SwiGLUFFN,
}


norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "rmsnorm": PT_RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def PT_init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, PT_LayerScale):
        module.reset_parameters()
    if isinstance(module, PT_PatchEmbed):
        module.reset_parameters()
    if isinstance(module, PT_RMSNorm):
        module.reset_parameters()


class PT_DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = (
            "separate"
        ),
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        kwargs = dict(img_size=img_size, patch_size=patch_size)
        kwargs.update(in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = PT_PatchEmbed(**kwargs, flatten_embedding=False)

        cls_token = torch.empty(1, 1, embed_dim, device=device)
        self.cls_token = nn.Parameter(cls_token)
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            shape = (1, n_storage_tokens, embed_dim)
            storage_tokens = torch.empty(*shape, device=device)
            self.storage_tokens = nn.Parameter(storage_tokens)
        suffix = " for rope new"
        logger.info(f"using base={pos_embed_rope_base}" + suffix)
        logger.info(f"using min_period={pos_embed_rope_min_period}" + suffix)
        logger.info(f"using max_period={pos_embed_rope_max_period}" + suffix)
        message = f"using normalize_coords={pos_embed_rope_normalize_coords}"
        logger.info(message + suffix)
        message = f"using shift_coords={pos_embed_rope_shift_coords}"
        logger.info(message + suffix)
        message = f"using rescale_coords={pos_embed_rope_rescale_coords}"
        logger.info(message + suffix)
        message = f"using jitter_coords={pos_embed_rope_jitter_coords}"
        logger.info(message + suffix)
        logger.info(f"using dtype={pos_embed_rope_dtype}" + suffix)
        kwargs = dict(embed_dim=embed_dim, num_heads=num_heads)
        kwargs.update(base=pos_embed_rope_base)
        kwargs.update(min_period=pos_embed_rope_min_period)
        kwargs.update(max_period=pos_embed_rope_max_period)
        kwargs.update(normalize_coords=pos_embed_rope_normalize_coords)
        kwargs.update(shift_coords=pos_embed_rope_shift_coords)
        kwargs.update(jitter_coords=pos_embed_rope_jitter_coords)
        kwargs.update(rescale_coords=pos_embed_rope_rescale_coords)
        kwargs.update(dtype=dtype_dict[pos_embed_rope_dtype], device=device)
        self.rope_embed = PT_RopePositionEmbedding(**kwargs)
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = []
        for i in range(depth):
            kwargs = dict(dim=embed_dim, num_heads=num_heads)
            kwargs.update(ffn_ratio=ffn_ratio_sequence[i], qkv_bias=qkv_bias)
            kwargs.update(proj_bias=proj_bias, ffn_bias=ffn_bias)
            kwargs.update(drop_path=drop_path_rate, norm_layer=norm_layer_cls)
            kwargs.update(act_layer=nn.GELU, ffn_layer=ffn_layer_cls)
            kwargs.update(init_values=layerscale_init)
            kwargs.update(mask_k_bias=mask_k_bias, device=device)
            blocks_list.append(PT_SelfAttentionBlock(**kwargs))

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and
        # mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        untie_local = untie_global_and_local_cls_norm
        self.untie_global_and_local_cls_norm = untie_local
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and
            # registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        mask_token = torch.empty(1, embed_dim, device=device)
        self.mask_token = nn.Parameter(mask_token)

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(PT_init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks=None
    ) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            mask_token = self.mask_token.to(x.dtype).unsqueeze(0)
            x = torch.where(masks.unsqueeze(-1), mask_token, x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            kwargs = dict(dtype=cls_token.dtype, device=cls_token.device)
            shape = (1, 0, cls_token.shape[-1])
            storage_tokens = torch.empty(*shape, **kwargs)

        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, storage_tokens.expand(B, -1, -1), x], dim=1)

        return x, (H, W)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        num_prefix_tokens = self.n_storage_tokens + 1
        untie_local = self.untie_global_and_local_cls_norm
        untie_any = self.untie_cls_and_patch_norms or untie_local
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if untie_any:
                prefix_tokens = x[:, :num_prefix_tokens]
                if untie_local and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(prefix_tokens)
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(prefix_tokens)
                else:
                    x_norm_cls_reg = self.norm(prefix_tokens)
                x_norm_patch = self.norm(x[:, num_prefix_tokens:])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, :num_prefix_tokens]
                x_norm_patch = x_norm[:, num_prefix_tokens:]
            features = {
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": x,
                "masks": masks,
            }
            output.append(features)
        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: int = 1
    ) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        if isinstance(n, int):
            blocks_to_take = range(total_block_len - n, total_block_len)
        else:
            blocks_to_take = n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        num_found, num_taken = len(output), len(blocks_to_take)
        message = f"only {num_found} / {num_taken} blocks found"
        assert num_found == num_taken, message
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        num_prefix_tokens = self.n_storage_tokens + 1
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    prefix_tokens = out[:, :num_prefix_tokens]
                    x_norm_cls_reg = self.cls_norm(prefix_tokens)
                    x_norm_patch = self.norm(out[:, num_prefix_tokens:])
                    tokens = (x_norm_cls_reg, x_norm_patch)
                    outputs_normed.append(torch.cat(tokens, dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1:num_prefix_tokens] for out in outputs]
        outputs = [out[:, num_prefix_tokens:] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self, *args, is_training: bool = False, **kwargs
    ) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def PT_vit_small(patch_size=16, **kwargs):
    config = dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=4)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model


def PT_vit_base(patch_size=16, **kwargs):
    config = dict(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model


def PT_vit_large(patch_size=16, **kwargs):
    config = dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model


def PT_vit_so400m(patch_size=16, **kwargs):
    config = dict(embed_dim=1152, depth=27, num_heads=18)
    config.update(ffn_ratio=3.777777778)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model


def PT_vit_huge2(patch_size=16, **kwargs):
    config = dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=4)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model


def PT_vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per
    head 64
    """
    config = dict(embed_dim=1536, depth=40, num_heads=24, ffn_ratio=4)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model


def PT_vit_7b(patch_size=16, **kwargs):
    config = dict(embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3)
    model = PT_DinoVisionTransformer(patch_size=patch_size, **config, **kwargs)
    return model
