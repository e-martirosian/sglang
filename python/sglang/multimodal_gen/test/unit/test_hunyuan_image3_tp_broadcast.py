"""Unit tests for HunyuanImage-3 tensor-parallel input broadcasts."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    ar_stage,
)
from sglang.multimodal_gen.runtime.models.dits import hunyuan_image3 as hi3_model
from sglang.test.test_utils import CustomTestCase


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def contiguous(self):
        return self

    def reshape(self, *_shape):
        return self


class _FakeTPGroup:
    world_size = 2

    def __init__(self):
        self.broadcast_calls = []

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))
        return tensor


class _ContiguousValue:
    def contiguous(self):
        return self


class _RecordingLayer:
    def __init__(self, kv_state):
        self.kv_state = kv_state
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return args[1], None, self.kv_state


class TestHunyuanImage3TPBroadcast(CustomTestCase):
    def test_static_broadcast_happens_outside_denoising_loop(self):
        source = textwrap.dedent(
            inspect.getsource(ar_stage.HunyuanImage3AR._forward_batched)
        )
        function = ast.parse(source).body[0]
        static_broadcast_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_broadcast_static_inputs"
        ]
        self.assertEqual(len(static_broadcast_calls), 1)

        denoising_loops = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and node.iter.args
            and isinstance(node.iter.args[0], ast.Call)
            and isinstance(node.iter.args[0].func, ast.Attribute)
            and node.iter.args[0].func.attr == "progress_bar"
        ]
        self.assertEqual(len(denoising_loops), 1)
        self.assertLess(
            static_broadcast_calls[0].lineno,
            denoising_loops[0].lineno,
            "request-static inputs must be broadcast before the denoising loop",
        )

    def test_static_inputs_are_broadcast_once(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        attention_mask = _FakeTensor((2, 1, 8, 8))
        cos = _FakeTensor((2, 8, 4))
        sin = _FakeTensor((2, 8, 4))

        with (
            patch.object(
                ar_stage, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._broadcast_static_inputs(attention_mask, (cos, sin))

        self.assertEqual(result, (attention_mask, (cos, sin)))
        self.assertEqual(
            tp_group.broadcast_calls,
            [(attention_mask, 0), (cos, 0), (sin, 0)],
        )

    def test_backbone_forward_only_broadcasts_dynamic_hidden_states(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        hidden_states = _FakeTensor((2, 3, 4))
        attention_mask = _FakeTensor((2, 1, 3, 3))
        cos = _FakeTensor((2, 3, 2))
        sin = _FakeTensor((2, 3, 2))
        output = MagicMock()
        output.shape = (6, 4)
        output.view.return_value = "reshaped-output"
        stage._cache_dit_adapter = None
        stage._cache_dit_controller = None
        stage.ar_model = SimpleNamespace(
            forward_block=MagicMock(return_value=output)
        )

        with (
            patch.object(
                ar_stage, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._backbone_forward(
                num_image_tokens=2,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                custom_pos_emb=(cos, sin),
                first_step=True,
            )

        self.assertEqual(result, "reshaped-output")
        self.assertEqual(tp_group.broadcast_calls, [(hidden_states, 0)])

    def test_disabled_cache_controller_uses_native_backbone(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        hidden_states = _FakeTensor((2, 3, 4))
        attention_mask = _FakeTensor((2, 1, 3, 3))
        cos = _FakeTensor((2, 3, 2))
        sin = _FakeTensor((2, 3, 2))
        output = MagicMock()
        output.shape = (6, 4)
        output.view.return_value = "native-output"
        stage._cache_dit_adapter = MagicMock(return_value=output)
        stage._cache_dit_controller = SimpleNamespace(enabled=False)
        stage.ar_model = SimpleNamespace(
            forward_block=MagicMock(return_value=output)
        )

        result = stage._backbone_forward(
            num_image_tokens=2,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            custom_pos_emb=(cos, sin),
            first_step=True,
        )

        self.assertEqual(result, "native-output")
        stage.ar_model.forward_block.assert_called_once()
        stage._cache_dit_adapter.assert_not_called()

    def test_cache_dit_and_teacache_are_mutually_exclusive(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        batch = SimpleNamespace(
            enable_teacache=True,
            sampling_params=SimpleNamespace(enable_cache_dit=True),
        )

        with self.assertRaisesRegex(ValueError, "Cache-DiT and TeaCache"):
            stage._maybe_enable_cache_dit(
                8,
                batch,
                SimpleNamespace(enable_breakable_cuda_graph=False),
                do_cfg=False,
            )

    def test_cache_adapter_shares_attention_metadata_and_preserves_cla_kv(self):
        layers = [
            _RecordingLayer("master-kv"),
            _RecordingLayer("follower-1-kv"),
            _RecordingLayer("follower-2-kv"),
        ]
        model = SimpleNamespace(
            config=SimpleNamespace(use_cla=True, cla_share_factor=3),
            layers=layers,
        )
        adapter = hi3_model.Hi3CacheBlockAdapter(model)
        hidden_states = _ContiguousValue()

        with patch.object(
            hi3_model,
            "create_hunyuan_image_attention_meta",
            return_value="attn-meta",
        ) as create_meta:
            result = adapter(
                hidden_states,
                "attention-mask",
                "rope",
                num_image_tokens=16,
                first_step=True,
            )

        self.assertIs(result, hidden_states)
        create_meta.assert_called_once_with("attention-mask", 16, True)
        self.assertIsNone(layers[0].calls[0][4])
        self.assertEqual(layers[0].calls[0][5], "attn-meta")
        self.assertEqual(layers[1].calls[0][4], "master-kv")
        self.assertEqual(layers[1].calls[0][5], "attn-meta")
        self.assertEqual(layers[2].calls[0][4], "master-kv")
        self.assertEqual(layers[2].calls[0][5], "attn-meta")
        with self.assertRaisesRegex(ValueError, "Bn_compute_blocks=0"):
            adapter.validate_cache_dit_config(SimpleNamespace(Bn_compute_blocks=1))

    def test_native_block_loop_preserves_cla_kv_for_all_followers(self):
        layers = [
            _RecordingLayer("master-kv"),
            _RecordingLayer("follower-1-kv"),
            _RecordingLayer("follower-2-kv"),
        ]
        model = hi3_model.HunyuanImage3Model.__new__(hi3_model.HunyuanImage3Model)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(use_cla=True, cla_share_factor=3),
        )
        object.__setattr__(model, "layers", layers)

        hi3_model.HunyuanImage3Model.forward_block(
            model,
            _ContiguousValue(),
            "attention-mask",
            "rope",
            attn_meta="attn-meta",
        )

        self.assertIsNone(layers[0].calls[0][4])
        self.assertEqual(layers[1].calls[0][4], "master-kv")
        self.assertEqual(layers[2].calls[0][4], "master-kv")


if __name__ == "__main__":
    unittest.main()
