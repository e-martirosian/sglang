"""Cache-DiT execution-contract tests for HunyuanImage-3."""

from types import SimpleNamespace

import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 as hunyuan_image3


class _MountedCacheDitBlocks(nn.Module):
    """Stand-in for cache-dit's one-element ModuleList wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attn_meta,
        attention_mask: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self.calls.append(
            {
                "attn_meta": attn_meta,
                "attention_mask": attention_mask,
                "custom_pos_emb": custom_pos_emb,
            }
        )
        return hidden_states + 1


class _NativeBackbone:
    def __init__(self) -> None:
        self.calls = []

    def forward_block(
        self,
        hidden_states,
        attention_mask,
        custom_pos_emb,
        *,
        num_image_tokens,
        first_step,
    ):
        self.calls.append(
            {
                "attention_mask": attention_mask,
                "custom_pos_emb": custom_pos_emb,
                "num_image_tokens": num_image_tokens,
                "first_step": first_step,
            }
        )
        return hidden_states + 2


def test_forward_uses_native_backbone_when_cache_dit_is_disabled():
    model = hunyuan_image3.HunyuanImage3ForCausalMM.__new__(
        hunyuan_image3.HunyuanImage3ForCausalMM
    )
    nn.Module.__init__(model)
    native_backbone = _NativeBackbone()
    model.model = native_backbone

    hidden_states = torch.zeros(3, 4)
    attention_mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    cos = torch.zeros(1, 3, 2)
    sin = torch.ones(1, 3, 2)

    output = model.forward(
        hidden_states,
        attention_mask=attention_mask,
        custom_pos_emb=(cos, sin),
        num_image_tokens=2,
        first_step=True,
    )

    assert torch.equal(output, hidden_states + 2)
    assert len(native_backbone.calls) == 1
    call = native_backbone.calls[0]
    assert call["attention_mask"] is attention_mask
    assert call["custom_pos_emb"][0] is cos
    assert call["custom_pos_emb"][1] is sin
    assert call["num_image_tokens"] == 2
    assert call["first_step"] is True


def test_forward_executes_cache_dit_mounted_blocks(monkeypatch):
    """The native forward accepts cache-dit's temporary block replacement."""
    model = hunyuan_image3.HunyuanImage3ForCausalMM.__new__(
        hunyuan_image3.HunyuanImage3ForCausalMM
    )
    nn.Module.__init__(model)
    model.model = SimpleNamespace(config=SimpleNamespace(use_cla=False))

    mounted_blocks = _MountedCacheDitBlocks()
    # Cache-DiT temporarily replaces `transformer_blocks` with exactly this
    # shape: a ModuleList containing one CachedBlocks_Pattern_3_4_5 wrapper.
    model.transformer_blocks = nn.ModuleList([mounted_blocks])
    model._sglang_cache_dit_adapter = object()

    attention_meta = object()
    monkeypatch.setattr(
        hunyuan_image3,
        "create_hunyuan_image_attention_meta",
        lambda attention_mask, num_image_tokens, first_step: attention_meta,
    )

    hidden_states = torch.zeros(3, 4)
    attention_mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    cos = torch.zeros(1, 3, 2)
    sin = torch.ones(1, 3, 2)

    output = model.forward(
        hidden_states,
        attention_mask=attention_mask,
        custom_pos_emb=(cos, sin),
        num_image_tokens=2,
        first_step=True,
    )

    assert torch.equal(output, hidden_states + 1)
    assert len(mounted_blocks.calls) == 1
    call = mounted_blocks.calls[0]
    assert call["attn_meta"] is attention_meta
    assert call["attention_mask"] is attention_mask
    assert call["custom_pos_emb"][0] is cos
    assert call["custom_pos_emb"][1] is sin
