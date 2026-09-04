import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeDBCacheConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self, **kwargs):
        return kwargs


class _FakeForwardPattern:
    # A class (not a SimpleNamespace instance) so it is a valid type in
    # annotations like List[ForwardPattern], matching the real Enum.
    Pattern_2 = "Pattern_2"
    Pattern_3 = "Pattern_3"


def _install_cache_dit_stub():
    cache_dit = types.ModuleType("cache_dit")
    cache_dit.enable_calls = []
    cache_dit.disable_calls = []
    cache_dit.refresh_calls = []
    cache_dit.steps_mask_calls = []

    def enable_cache(target, **kwargs):
        cache_dit.enable_calls.append({"target": target, **kwargs})

    def disable_cache(target):
        cache_dit.disable_calls.append(target)

    def refresh_context(transformer, cache_config, verbose=False):
        cache_dit.refresh_calls.append(
            {
                "transformer": transformer,
                "cache_config": cache_config,
                "verbose": verbose,
            }
        )

    def steps_mask(
        *, mask_policy, total_steps, compute_bins=None, cache_bins=None
    ):
        call = {"mask_policy": mask_policy, "total_steps": total_steps}
        if compute_bins is not None:
            call["compute_bins"] = list(compute_bins)
        if cache_bins is not None:
            call["cache_bins"] = list(cache_bins)
        cache_dit.steps_mask_calls.append(call)
        return [1] * total_steps

    cache_dit.enable_cache = enable_cache
    cache_dit.disable_cache = disable_cache
    cache_dit.refresh_context = refresh_context
    cache_dit.steps_mask = steps_mask
    cache_dit.BlockAdapter = types.SimpleNamespace
    cache_dit.DBCacheConfig = _FakeDBCacheConfig
    cache_dit.ForwardPattern = _FakeForwardPattern
    cache_dit.ParamsModifier = object
    cache_dit.TaylorSeerCalibratorConfig = object

    block_adapters = types.ModuleType("cache_dit.caching.block_adapters")

    class _FakeBlockAdapterRegister:
        supported = True

        @classmethod
        def is_supported(cls, _transformer):
            return cls.supported

    block_adapters.BlockAdapterRegister = _FakeBlockAdapterRegister

    parallelism = types.ModuleType("cache_dit.parallelism")
    parallelism.ParallelismBackend = object
    parallelism.ParallelismConfig = object

    return {
        "cache_dit": cache_dit,
        "cache_dit.caching.block_adapters": block_adapters,
        "cache_dit.parallelism": parallelism,
    }


def _install_sglang_dependency_stubs():
    sglang = types.ModuleType("sglang")
    multimodal_gen = types.ModuleType("sglang.multimodal_gen")
    envs = types.ModuleType("sglang.multimodal_gen.envs")
    runtime = types.ModuleType("sglang.multimodal_gen.runtime")
    distributed = types.ModuleType("sglang.multimodal_gen.runtime.distributed")
    parallel_state = types.ModuleType(
        "sglang.multimodal_gen.runtime.distributed.parallel_state"
    )
    utils = types.ModuleType("sglang.multimodal_gen.runtime.utils")
    logging_utils = types.ModuleType(
        "sglang.multimodal_gen.runtime.utils.logging_utils"
    )

    parallel_state.get_ring_parallel_world_size = lambda: 1
    parallel_state.get_tp_world_size = lambda: 1
    parallel_state.get_ulysses_parallel_world_size = lambda: 1
    parallel_state.get_dit_group = lambda: None

    envs.SGLANG_CACHE_DIT_ENABLED = False
    envs.SGLANG_CACHE_DIT_FN = 1
    envs.SGLANG_CACHE_DIT_BN = 0
    envs.SGLANG_CACHE_DIT_WARMUP = 4
    envs.SGLANG_CACHE_DIT_RDT = 0.24
    envs.SGLANG_CACHE_DIT_MC = 3
    envs.SGLANG_CACHE_DIT_TAYLORSEER = False
    envs.SGLANG_CACHE_DIT_TS_ORDER = 1
    envs.SGLANG_CACHE_DIT_SCM_PRESET = "none"
    envs.SGLANG_CACHE_DIT_SCM_POLICY = "dynamic"
    envs.SGLANG_CACHE_DIT_SCM_COMPUTE_BINS = None
    envs.SGLANG_CACHE_DIT_SCM_CACHE_BINS = None

    class _FakeLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def warning_once(self, *_args, **_kwargs):
            pass

    logging_utils.init_logger = lambda _name: _FakeLogger()

    return {
        "sglang": sglang,
        "sglang.multimodal_gen": multimodal_gen,
        "sglang.multimodal_gen.envs": envs,
        "sglang.multimodal_gen.runtime": runtime,
        "sglang.multimodal_gen.runtime.distributed": distributed,
        "sglang.multimodal_gen.runtime.distributed.parallel_state": parallel_state,
        "sglang.multimodal_gen.runtime.utils": utils,
        "sglang.multimodal_gen.runtime.utils.logging_utils": logging_utils,
    }


def _install_torch_stub():
    torch = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    torch_dist = types.ModuleType("torch.distributed")

    class _FakeModule:
        pass

    class _FakeProcessGroup:
        pass

    class _FakeReduceOp:
        AVG = "AVG"

    torch_dist.all_reduce_calls = []

    def all_reduce(tensor, *, op, group):
        torch_dist.all_reduce_calls.append(
            {"tensor": tensor, "op": op, "group": group}
        )

    torch_nn.Module = _FakeModule
    torch_dist.ProcessGroup = _FakeProcessGroup
    torch_dist.ReduceOp = _FakeReduceOp
    torch_dist.all_reduce = all_reduce
    torch.distributed = torch_dist
    torch.nn = torch_nn
    torch.stack = lambda tensors: list(tensors)

    return {
        "torch": torch,
        "torch.nn": torch_nn,
        "torch.distributed": torch_dist,
    }


def _import_module_with_stub():
    stub_modules = _install_cache_dit_stub()
    stub_modules.update(_install_sglang_dependency_stubs())
    stub_modules.update(_install_torch_stub())
    module_path = (
        Path(__file__).resolve().parents[2]
        / "runtime"
        / "cache"
        / "cache_dit_integration.py"
    )
    with patch.dict(sys.modules, stub_modules):
        spec = importlib.util.spec_from_file_location(
            "test_cache_dit_integration_target", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class TestCacheDitParallelSimilarity(unittest.TestCase):
    def test_reduces_similarity_statistics_in_one_collective(self):
        module = _import_module_with_stub()
        group = object()

        mean_diff, mean_t1 = module._all_reduce_mean_pair(0.25, 2.0, group)

        self.assertEqual((mean_diff, mean_t1), (0.25, 2.0))
        self.assertEqual(
            module.dist.all_reduce_calls,
            [
                {
                    "tensor": [0.25, 2.0],
                    "op": "AVG",
                    "group": group,
                }
            ],
        )


class TestCacheDitRefreshContext(unittest.TestCase):
    def test_refresh_context_without_scm_preset_skips_steps_mask(self):
        module = _import_module_with_stub()
        module.refresh_context_on_transformer(
            transformer="transformer",
            num_inference_steps=50,
            scm_preset=None,
            verbose=True,
        )

        self.assertEqual(module.cache_dit.steps_mask_calls, [])
        self.assertEqual(len(module.cache_dit.refresh_calls), 1)
        self.assertEqual(
            module.cache_dit.refresh_calls[0]["cache_config"],
            {
                "num_inference_steps": 50,
                "steps_computation_mask": None,
                "steps_computation_policy": None,
            },
        )

    def test_refresh_context_with_scm_preset_uses_steps_mask(self):
        module = _import_module_with_stub()
        module.refresh_context_on_transformer(
            transformer="transformer",
            num_inference_steps=8,
            scm_preset="fast",
        )

        self.assertEqual(
            module.cache_dit.steps_mask_calls,
            [{"mask_policy": "fast", "total_steps": 8}],
        )
        self.assertEqual(
            module.cache_dit.refresh_calls[0]["cache_config"],
            {
                "num_inference_steps": 8,
                "steps_computation_mask": [1] * 8,
                "steps_computation_policy": "fast",
            },
        )

    def test_dual_refresh_without_scm_preset_skips_steps_mask(self):
        module = _import_module_with_stub()
        module.refresh_context_on_dual_transformer(
            transformer="transformer",
            transformer_2="transformer_2",
            num_high_noise_steps=12,
            num_low_noise_steps=6,
            scm_preset=None,
        )

        self.assertEqual(module.cache_dit.steps_mask_calls, [])
        self.assertEqual(len(module.cache_dit.refresh_calls), 2)
        self.assertEqual(
            module.cache_dit.refresh_calls[0]["cache_config"],
            {
                "num_inference_steps": 12,
                "steps_computation_mask": None,
                "steps_computation_policy": None,
            },
        )
        self.assertEqual(
            module.cache_dit.refresh_calls[1]["cache_config"],
            {
                "num_inference_steps": 6,
                "steps_computation_mask": None,
                "steps_computation_policy": None,
            },
        )


def _make_transformer(class_name, layers=None):
    transformer = type(class_name, (), {})()
    if layers is not None:
        transformer.layers = layers
    return transformer


class TestBuildCustomBlockAdapter(unittest.TestCase):
    def test_builds_adapter_for_registered_class(self):
        module = _import_module_with_stub()
        blocks = ["block_0", "block_1"]
        transformer = _make_transformer("ErnieImageTransformer2DModel", blocks)

        adapter = module._build_custom_block_adapter(transformer, has_separate_cfg=True)

        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.blocks, blocks)
        self.assertEqual(adapter.forward_pattern, "Pattern_3")
        self.assertTrue(adapter.has_separate_cfg)

    def test_returns_none_for_unknown_class(self):
        module = _import_module_with_stub()
        transformer = _make_transformer("SomeUnregisteredTransformer", ["b0"])

        self.assertIsNone(module._build_custom_block_adapter(transformer))

    def test_raises_when_blocks_attr_missing(self):
        module = _import_module_with_stub()
        transformer = _make_transformer("ErnieImageTransformer2DModel")

        with self.assertRaises(ValueError):
            module._build_custom_block_adapter(transformer)

    def test_has_separate_cfg_follows_runtime(self):
        # No model pins the mode; has_separate_cfg always follows the run's CFG mode
        # (Krea-2 Raw -> True, Krea-2 Turbo -> False).
        module = _import_module_with_stub()
        blocks = ["block_0", "block_1"]

        transformer_raw = _make_transformer("Krea2Transformer2DModel")
        transformer_raw.transformer_blocks = blocks
        adapter_raw = module._build_custom_block_adapter(
            transformer_raw, has_separate_cfg=True
        )
        self.assertEqual(adapter_raw.blocks, blocks)
        self.assertEqual(adapter_raw.forward_pattern, "Pattern_3")
        self.assertTrue(adapter_raw.has_separate_cfg)

        transformer_turbo = _make_transformer("Krea2Transformer2DModel")
        transformer_turbo.transformer_blocks = blocks
        adapter_turbo = module._build_custom_block_adapter(
            transformer_turbo, has_separate_cfg=False
        )
        self.assertFalse(adapter_turbo.has_separate_cfg)

    def test_minimax_h3_uses_main_blocks_with_hidden_state_pattern(self):
        module = _import_module_with_stub()
        blocks = ["block_0", "block_1"]
        transformer = _make_transformer("MiniMaxH3DiTModel")
        transformer.blocks = blocks

        adapter = module._build_custom_block_adapter(transformer)

        self.assertEqual(adapter.blocks, blocks)
        self.assertEqual(adapter.forward_pattern, "Pattern_3")
        self.assertFalse(adapter.has_separate_cfg)

    def test_custom_adapter_is_retained_until_disable(self):
        module = _import_module_with_stub()
        module.BlockAdapterRegister.supported = False
        transformer = _make_transformer("MiniMaxH3DiTModel")
        transformer.blocks = ["block_0"]
        config = module.CacheDitConfig(enabled=True, num_inference_steps=50)

        returned = module.enable_cache_on_transformer(transformer, config)

        self.assertIs(returned, transformer)
        adapter = transformer._sglang_cache_dit_adapter
        self.assertIs(module.cache_dit.enable_calls[0]["target"], adapter)

        self.assertIs(module.disable_cache_on_transformer(transformer), transformer)
        self.assertEqual(module.cache_dit.disable_calls, [adapter])
        self.assertFalse(hasattr(transformer, "_sglang_cache_dit_adapter"))

    def test_hunyuan_image3_uses_pattern_3_facade_blocks(self):
        module = _import_module_with_stub()
        transformer = _make_transformer("HunyuanImage3ForCausalMM")
        transformer.transformer_blocks = ["block_0", "block_1"]

        adapter = module._build_custom_block_adapter(transformer)

        self.assertEqual(adapter.blocks, transformer.transformer_blocks)
        self.assertEqual(adapter.forward_pattern, "Pattern_3")

    def test_hunyuan_image3_custom_adapter_wins_prefix_registry_collision(self):
        module = _import_module_with_stub()
        module.BlockAdapterRegister.supported = True
        transformer = _make_transformer("HunyuanImage3ForCausalMM")
        transformer.transformer_blocks = ["block_0", "block_1"]
        config = module.CacheDitConfig(enabled=True, num_inference_steps=8)

        returned = module.enable_cache_on_transformer(transformer, config)

        self.assertIs(returned, transformer)
        adapter = transformer._sglang_cache_dit_adapter
        self.assertIs(module.cache_dit.enable_calls[0]["target"], adapter)
        self.assertEqual(adapter.blocks, transformer.transformer_blocks)
        self.assertEqual(adapter.forward_pattern, "Pattern_3")


class TestCacheDitController(unittest.TestCase):
    def test_warmup_unmounts_cache_from_previous_request(self):
        module = _import_module_with_stub()
        module.BlockAdapterRegister.supported = False
        transformer = _make_transformer("HunyuanImage3ForCausalMM")
        transformer.transformer_blocks = ["block_0"]
        server_args = types.SimpleNamespace(enable_breakable_cuda_graph=False)
        enabled_params = types.SimpleNamespace(
            enable_cache_dit=True, cache_dit_params=None
        )
        controller = module.CacheDitController(transformer, server_args)

        controller.configure(
            12,
            types.SimpleNamespace(
                sampling_params=enabled_params,
                is_warmup=False,
            ),
        )
        controller.configure(
            12,
            types.SimpleNamespace(
                sampling_params=enabled_params,
                is_warmup=True,
            ),
        )

        self.assertFalse(controller.enabled)
        self.assertIsNone(controller.active_key)
        self.assertEqual(len(module.cache_dit.disable_calls), 1)
        self.assertEqual(module.cache_dit.refresh_calls, [])

    def test_mount_refresh_and_request_opt_out(self):
        module = _import_module_with_stub()
        module.BlockAdapterRegister.supported = False
        transformer = _make_transformer("HunyuanImage3ForCausalMM")
        transformer.transformer_blocks = ["block_0"]
        server_args = types.SimpleNamespace(enable_breakable_cuda_graph=False)
        enabled_params = types.SimpleNamespace(
            enable_cache_dit=True, cache_dit_params=None
        )
        enabled_batch = types.SimpleNamespace(
            sampling_params=enabled_params, is_warmup=False
        )
        controller = module.CacheDitController(transformer, server_args)

        controller.configure(12, enabled_batch)
        self.assertTrue(controller.enabled)
        self.assertEqual(len(module.cache_dit.enable_calls), 1)

        controller.configure(12, enabled_batch)
        self.assertEqual(len(module.cache_dit.refresh_calls), 1)

        disabled_batch = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                enable_cache_dit=False, cache_dit_params=None
            ),
            is_warmup=False,
        )
        controller.configure(12, disabled_batch)
        self.assertFalse(controller.enabled)
        self.assertEqual(len(module.cache_dit.disable_calls), 1)

    def test_refresh_preserves_static_scm_policy(self):
        module = _import_module_with_stub()
        module.BlockAdapterRegister.supported = False
        transformer = _make_transformer("HunyuanImage3ForCausalMM")
        transformer.transformer_blocks = ["block_0"]
        server_args = types.SimpleNamespace(enable_breakable_cuda_graph=False)
        batch = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                enable_cache_dit=True,
                cache_dit_params={
                    "scm_preset": "fast",
                    "scm_policy": "static",
                },
            ),
            is_warmup=False,
        )
        controller = module.CacheDitController(transformer, server_args)

        controller.configure(8, batch)
        controller.configure(12, batch)

        mounted_config = module.cache_dit.enable_calls[0]["cache_config"]
        self.assertEqual(
            mounted_config.kwargs["steps_computation_policy"], "static"
        )
        refreshed_config = module.cache_dit.refresh_calls[0]["cache_config"]
        self.assertEqual(refreshed_config["steps_computation_policy"], "static")
        self.assertEqual(len(refreshed_config["steps_computation_mask"]), 12)

    def test_refresh_rebuilds_custom_scm_mask_for_new_step_count(self):
        module = _import_module_with_stub()
        module.BlockAdapterRegister.supported = False
        transformer = _make_transformer("HunyuanImage3ForCausalMM")
        transformer.transformer_blocks = ["block_0"]
        server_args = types.SimpleNamespace(enable_breakable_cuda_graph=False)
        batch = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                enable_cache_dit=True,
                cache_dit_params={
                    "scm_compute_bins": [4, 4],
                    "scm_cache_bins": [2, 2],
                    "scm_policy": "static",
                },
            ),
            is_warmup=False,
        )
        controller = module.CacheDitController(transformer, server_args)

        controller.configure(8, batch)
        controller.configure(12, batch)

        refreshed_config = module.cache_dit.refresh_calls[0]["cache_config"]
        self.assertEqual(len(refreshed_config["steps_computation_mask"]), 12)
        self.assertEqual(refreshed_config["steps_computation_policy"], "static")
        self.assertEqual(
            module.cache_dit.steps_mask_calls[-1],
            {
                "mask_policy": "medium",
                "total_steps": 12,
                "compute_bins": [4, 4],
                "cache_bins": [2, 2],
            },
        )


if __name__ == "__main__":
    unittest.main()
