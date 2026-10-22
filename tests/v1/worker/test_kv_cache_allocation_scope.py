# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm.v1.hisparse.binding as hisparse_binding
import vllm.v1.worker.gpu.attn_utils as attn_utils
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from vllm.v1.worker.gpu_worker import Worker


class _AllocationScope(AbstractContextManager):
    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *args: Any) -> None:
        assert self.active
        self.active = False


@pytest.mark.parametrize("hisparse", [False, True])
def test_mrv2_kv_pool_only_wraps_backing_allocation(monkeypatch, hisparse) -> None:
    scope = _AllocationScope()
    kv_caches = {"layer": torch.empty(0)}

    def allocate(*args, **kwargs):
        assert scope.active
        return kv_caches

    def bind(*args, **kwargs):
        assert not scope.active

    monkeypatch.setattr(attn_utils, "allocate_kv_cache", allocate)
    monkeypatch.setattr(attn_utils, "bind_kv_cache_to_layers", bind)
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})

    hisparse_bindings = []
    if hisparse:
        host_pool = SimpleNamespace()

        def bind_hisparse(**kwargs):
            hisparse_bindings.append(kwargs)
            assert scope.active
            assert kwargs["kv_caches"] is kv_caches
            assert kwargs["host_pool"] is host_pool
            return [
                SimpleNamespace(
                    view=SimpleNamespace(cache=torch.empty(1, 1, 1), block_size=1),
                    runtime=SimpleNamespace(),
                )
            ]

        monkeypatch.setattr(
            hisparse_binding,
            "HiSparseHostPool",
            lambda vllm_config, kv_cache_config: host_pool,
        )
        monkeypatch.setattr(hisparse_binding, "allocate_hisparse_kv_caches", allocate)
        monkeypatch.setattr(hisparse_binding, "bind_hisparse_kv_caches", bind_hisparse)

    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            hisparse_config=object() if hisparse else None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
    )
    result = attn_utils.init_kv_cache(
        {},
        SimpleNamespace(kv_cache_groups=[]),
        torch.device("cpu"),
        [],
        config,
        kv_cache_allocation_context=scope,
        block_tables=SimpleNamespace(),
    )

    assert len(hisparse_bindings) == int(hisparse)
    assert result is kv_caches
    assert not scope.active


def test_mrv1_kv_pool_only_wraps_backing_allocation(monkeypatch) -> None:
    scope = _AllocationScope()
    kv_caches = {"layer": torch.empty(0)}

    def allocate(*args, **kwargs):
        assert scope.active
        return kv_caches

    def bind(*args, **kwargs):
        assert not scope.active

    monkeypatch.setattr(gpu_model_runner, "allocate_kv_cache", allocate)
    monkeypatch.setattr(gpu_model_runner, "bind_kv_cache", bind)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        shared_kv_cache_layers={},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_caches=[],
        attn_groups=[],
        speculative_config=None,
    )
    result = gpu_model_runner.GPUModelRunner.initialize_kv_cache_tensors(
        runner,
        SimpleNamespace(kv_cache_groups=[]),
        [],
        kv_cache_allocation_context=scope,
    )

    assert result is kv_caches
    assert not scope.active


def test_kv_wake_does_not_run_model_runner_recovery() -> None:
    model = torch.nn.Module()
    model.register_buffer("_k_scale", torch.tensor(0.5))
    model.register_buffer("_v_scale", torch.tensor(0.25))

    class Runner:
        def __init__(self) -> None:
            self.model = model
            self.layout_tensors = tuple(torch.tensor([i]) for i in range(5))
            self.recovery_calls = 0

        def post_kv_cache_wake_up(self) -> None:
            self.recovery_calls += 1
            self.model.get_buffer("_k_scale").fill_(1.0)
            self.model.get_buffer("_v_scale").fill_(1.0)
            self.layout_tensors = tuple(torch.tensor([i]) for i in range(5))

    runner = Runner()
    worker = cast(
        Worker,
        SimpleNamespace(
            sleep_mode_backend=SimpleNamespace(resume=lambda tags: None),
            _sleep_saved_buffers={},
            _sleep_saved_draft_buffers={},
            model_runner=runner,
            synchronize_device=lambda: None,
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enable_nccl_comm_suspend=False)
            ),
        ),
    )
    layout_tensors = runner.layout_tensors
    layout_ptrs = tuple(t.data_ptr() for t in layout_tensors)

    Worker.wake_up(worker, tags=["kv_cache"])

    assert runner.recovery_calls == 0
    assert model.get_buffer("_k_scale").item() == 0.5
    assert model.get_buffer("_v_scale").item() == 0.25
    assert all(
        actual is expected
        for actual, expected in zip(runner.layout_tensors, layout_tensors, strict=True)
    )
    assert tuple(t.data_ptr() for t in runner.layout_tensors) == layout_ptrs


def test_imported_mrv2_kv_is_bound_without_allocating(monkeypatch) -> None:
    """An IPC consumer binds its own layers to the externally owned KV views."""
    caches = {"layer": torch.empty(1)}
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
    )
    forward_context = {"layer": object()}
    bound = {}

    def allocate(*args, **kwargs):
        pytest.fail("An IPC consumer must not allocate a second KV cache")

    def bind(views, layers, *args, **kwargs):
        bound.update(views)
        assert layers is forward_context

    monkeypatch.setattr(attn_utils, "allocate_kv_cache", allocate)
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})
    monkeypatch.setattr(attn_utils, "map_kv_caches_to_kernel_blocks", lambda x, _: x)
    monkeypatch.setattr(attn_utils, "bind_kv_cache_to_layers", bind)
    result = attn_utils.init_kv_cache(
        forward_context,
        SimpleNamespace(kv_cache_groups=[]),
        torch.device("cpu"),
        [],
        config,
        shared_kv_caches=caches,
    )
    assert result is caches
    assert bound["layer"] is caches["layer"]


@pytest.mark.parametrize("total,fraction", [(20, 0.5), (64, 0.3), (120, 0.7)])
def test_paired_cu_masks_partition_all_cu_pairs(total, fraction):
    """Each CU belongs to one role and adjacent CU pairs stay together."""
    from vllm.v1.rapidserve.resources import partition_cus

    prefill, decode = partition_cus(total, fraction)
    for cu in range(total):
        p = bool(prefill[cu // 32] & (1 << (cu % 32)))
        d = bool(decode[cu // 32] & (1 << (cu % 32)))
        assert p != d
        if cu % 2:
            assert p == bool(prefill[(cu - 1) // 32] & (1 << ((cu - 1) % 32)))
    assert sum(word.bit_count() for word in prefill + decode) == total


def test_paired_model_import_restores_host_tensors(monkeypatch):
    """Meta construction must not leave host scales or CPU buffers uninitialized."""
    from vllm.v1.rapidserve import ipc

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(0.25))
            self.host_scale = torch.tensor(0.5)

    source = Model()
    manifest = ipc.TensorViews([], {}, "test")
    monkeypatch.setattr(ipc.TensorViews, "export", lambda _: manifest)
    exported = ipc.export_model(source)
    monkeypatch.setattr(ipc, "validate_shared_model", lambda _: None)
    monkeypatch.setattr(ipc, "initialize_model", lambda _: Model())
    monkeypatch.setattr(
        ipc.TensorViews,
        "import_tensors",
        lambda self, _: {name: t.clone() for name, t in self.host_tensors.items()},
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        model_config=SimpleNamespace(dtype=torch.float32, is_moe=False),
    )
    imported = ipc.import_model(config, exported)
    assert imported.scale.device.type == imported.host_scale.device.type == "cpu"
    assert imported.scale.item() == 0.25
    assert imported.host_scale.item() == 0.5
    source.host_scale.fill_(9)
    assert imported.host_scale.item() == 0.5


def test_paired_model_import_preserves_distinct_dtypes_on_shared_storage(monkeypatch):
    """Equal-sized FP8 and byte views must not become the same Parameter."""
    from vllm.v1.rapidserve import ipc

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fp8 = torch.nn.Parameter(
                torch.empty(4, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.bytes = torch.nn.Parameter(
                torch.empty(4, dtype=torch.uint8), requires_grad=False
            )
            self.tied = self.fp8

    fp8 = torch.ones(4).to(torch.float8_e4m3fn)
    byte_view = fp8.view(torch.uint8)
    manifest = ipc.TensorViews([], {}, "test")
    monkeypatch.setattr(ipc, "validate_shared_model", lambda _: None)
    monkeypatch.setattr(ipc, "initialize_model", lambda _: Model())
    monkeypatch.setattr(
        ipc.TensorViews,
        "import_tensors",
        lambda *_: {
            "parameter:fp8": fp8,
            "parameter:bytes": byte_view,
            "parameter:tied": fp8,
        },
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        model_config=SimpleNamespace(dtype=torch.float32, is_moe=False),
    )
    imported = ipc.import_model(config, manifest)
    assert imported.fp8.dtype == torch.float8_e4m3fn
    assert imported.bytes.dtype == torch.uint8
    assert imported.fp8 is imported.tied
    assert imported.fp8 is not imported.bytes
    assert imported.fp8.data_ptr() == imported.bytes.data_ptr()
    torch.testing.assert_close(imported.fp8.float(), fp8.float())


def test_paired_moe_import_rebuilds_kernel_without_repacking_weights(monkeypatch):
    """IPC import preserves runtime-format expert storage and initializes execution."""
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.utils import is_weights_pre_processed
    from vllm.v1.rapidserve import ipc

    class Method(UnquantizedFusedMoEMethod):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.supports_pre_processed_weights = True
            self.moe_kernel = None

        def _init_moe_kernel(self, layer):
            assert is_weights_pre_processed()
            assert not layer.weight.is_meta
            self.moe_kernel = lambda x: x @ layer.weight

        def _maybe_pad_weight(self, weight):
            pytest.fail("Imported expert weights must not be padded or repacked")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 3))
            self.quant_method = Method()

    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    original = weight.clone()
    manifest = ipc.TensorViews([], {}, "test")
    monkeypatch.setattr(ipc, "validate_shared_model", lambda _: None)
    monkeypatch.setattr(ipc, "initialize_model", lambda _: Model())
    monkeypatch.setattr(
        ipc.TensorViews, "import_tensors", lambda *_: {"parameter:weight": weight}
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_moe=True,
            word_embeddings_untied_by_checkpoint=False,
            quantization=None,
        ),
    )
    imported = ipc.import_model(config, manifest)
    assert imported.weight.data_ptr() == weight.data_ptr()
    assert imported.weight.stride() == weight.stride()
    torch.testing.assert_close(weight, original)
    torch.testing.assert_close(
        imported.quant_method.moe_kernel(torch.ones(2)), original.sum(dim=0)
    )
    assert not is_weights_pre_processed()


def test_paired_fp8_import_restores_consumed_slots_and_shared_mla_weights(monkeypatch):
    """Runtime FP8/MLA tensors stay shared while consumed scale slots disappear."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
    from vllm.model_executor.utils import is_weights_pre_processed
    from vllm.v1.rapidserve import ipc

    class Method(Fp8MoEMethod):
        def __init__(self):
            pass

        def _init_moe_kernel(self, layer):
            assert is_weights_pre_processed()
            assert layer.w13_weight.is_shuffled
            assert layer.w13_input_scale is None
            assert not hasattr(layer, "k_scale")
            self.moe_kernel = lambda x: x @ layer.w13_weight

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.empty(2, 2), requires_grad=False)
            self.w13_input_scale = torch.nn.Parameter(torch.empty(1))
            self.k_scale = torch.nn.Parameter(torch.empty(1))
            self.W_UK_T = torch.empty(2, 2)
            self.quant_method = Method()

    source = Model()
    source.w13_weight.is_shuffled = True
    source.w13_input_scale = None
    del source.k_scale
    manifest = ipc.TensorViews([], {}, "test")
    monkeypatch.setattr(ipc.TensorViews, "export", lambda _: manifest)
    ipc.export_model(source)
    monkeypatch.setattr(ipc, "validate_shared_model", lambda _: None)
    monkeypatch.setattr(ipc, "initialize_model", lambda _: Model())
    monkeypatch.setattr(
        ipc.TensorViews, "import_tensors", lambda self, _: self.host_tensors
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_moe=True,
            word_embeddings_untied_by_checkpoint=False,
            quantization="fp8",
        ),
    )
    imported = ipc.import_model(config, manifest)
    assert (
        imported.w13_weight.data_ptr()
        == manifest.host_tensors["parameter:w13_weight"].data_ptr()
    )
    assert imported.W_UK_T is manifest.host_tensors["attribute:.W_UK_T"]
    torch.testing.assert_close(
        imported.quant_method.moe_kernel(torch.ones(2)), imported.w13_weight.sum(dim=0)
    )


def test_paired_load_consumes_ipc_manifest_before_compilation(monkeypatch):
    """Compile cache hashing accepts only serializable additional configuration."""
    import json
    from unittest.mock import Mock

    from vllm.v1.rapidserve import worker as paired_worker

    manifest = paired_worker.TensorViews([], {}, "test")
    config = SimpleNamespace(
        additional_config={"rapidserve_role": "decode", "rapidserve_weights": manifest}
    )
    worker = paired_worker.SharedModelWorker.__new__(paired_worker.SharedModelWorker)
    worker.vllm_config = config
    worker.model_runner = SimpleNamespace(
        jit_warmup_registry=SimpleNamespace(activate=_AllocationScope),
        initialize_model_components=Mock(),
    )
    imported = torch.nn.Module()
    load = Mock(return_value=imported)
    monkeypatch.setattr(paired_worker, "import_model", load)
    monkeypatch.setattr(paired_worker, "set_torch_threads_for_runtime", lambda: None)
    worker.load_model()
    load.assert_called_once_with(config, manifest)
    assert worker.model_runner.model is imported
    assert json.loads(json.dumps(config.additional_config)) == {
        "rapidserve_role": "decode"
    }


def test_paired_deepseek_router_uses_imported_correction_bias(monkeypatch):
    """The non-module router must not retain its constructor's meta parameter."""
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopKRouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.v1.rapidserve import ipc

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Module()
            self.gate.bias = torch.nn.Parameter(torch.empty(4), requires_grad=False)
            self.experts = MoERunner.__new__(MoERunner)
            torch.nn.Module.__init__(self.experts)
            self.experts.routed_experts = torch.nn.Module()
            self.experts.routed_experts.e_score_correction_bias = self.gate.bias
            self.experts.router = GroupedTopKRouter(
                2, 4, 2, 1, e_score_correction_bias=self.gate.bias
            )

    source = Model()
    source.gate.bias.data.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    manifest = ipc.TensorViews([], {}, "test")
    monkeypatch.setattr(ipc.TensorViews, "export", lambda _: manifest)
    ipc.export_model(source)
    monkeypatch.setattr(ipc, "validate_shared_model", lambda _: None)
    monkeypatch.setattr(ipc, "initialize_model", lambda _: Model())
    monkeypatch.setattr(
        ipc.TensorViews, "import_tensors", lambda self, _: self.host_tensors
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_moe=True,
            word_embeddings_untied_by_checkpoint=False,
            quantization=None,
        ),
    )
    imported = ipc.import_model(config, manifest)
    bias = imported.experts.router.e_score_correction_bias
    assert bias is imported.experts.routed_experts.e_score_correction_bias
    assert not bias.is_meta
    torch.testing.assert_close(bias, source.gate.bias)


def test_paired_mla_finalize_does_not_recompute_imported_absorbed_weights():
    """D must not dequantize/repack projections already finalized by P."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.utils import weights_already_processed

    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.W_UK_T = torch.ones(2, 2)
    layer.W_UV = torch.zeros(2, 2)
    absorbed = layer.W_UK_T
    with weights_already_processed():
        layer.process_weights_after_loading(torch.bfloat16)
    assert layer.W_UK_T is absorbed


def test_paired_shutdown_collects_imported_model_cycles(monkeypatch):
    """MoE model cycles must release IPC views before the owner is closed."""
    from weakref import ref

    from vllm.v1.rapidserve.worker import SharedModelWorker

    worker = SharedModelWorker.__new__(SharedModelWorker)
    worker.streams = {}
    worker.default_stream = object()
    model = torch.nn.Module()
    model.parent = model
    reference = ref(model)
    worker.model = model
    del model
    monkeypatch.setattr(Worker, "shutdown", lambda self: delattr(self, "model"))
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _: None)
    worker.shutdown()
    assert reference() is None


def test_paired_shutdown_releases_captured_ipc_weights(monkeypatch):
    """An externally retained compiled model must release its IPC allocations."""
    from weakref import ref

    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.v1.rapidserve.worker import SharedModelWorker

    worker = SharedModelWorker.__new__(SharedModelWorker)
    worker.streams = {}
    worker.default_stream = object()
    worker.vllm_config = object()

    class Model(torch.nn.Module, TorchCompileWithNoGuardsWrapper):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, x):
            return x

    model = Model()
    model.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.aot_compiled_fn = SimpleNamespace(weight=model.weight)
    model._compiled_callable = SimpleNamespace(weight=model.weight)
    retained_weight = model.weight
    model.register_buffer("cache", torch.zeros(2))
    wrapper = CUDAGraphWrapper.__new__(CUDAGraphWrapper)
    wrapper.vllm_config = worker.vllm_config
    wrapper.concrete_cudagraph_entries = {"test": SimpleNamespace(tensor=model.cache)}
    monkeypatch.setattr(CUDAGraphWrapper, "_all_instances", [wrapper])
    model.W_UK_T = torch.ones(2)
    model.experts = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(model.experts)
    model.experts.router = SimpleNamespace(e_score_correction_bias=model.weight)
    router = model.experts.router
    references = [ref(model.cache), ref(model.W_UK_T)]
    worker.model_runner = SimpleNamespace(model=model)
    monkeypatch.setattr(
        Worker, "shutdown", lambda self: delattr(self.model_runner, "model")
    )
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    worker.shutdown()
    assert not wrapper.concrete_cudagraph_entries
    assert all(reference() is None for reference in references)
    assert retained_weight.numel() == 0
    assert router.e_score_correction_bias is None
    assert model.aot_compiled_fn is None and model._compiled_callable is None
    assert not list(model.parameters()) and not list(model.buffers())


def test_paired_equivalent_cu_policy_reuses_stream_and_graph(monkeypatch):
    """Fractions rounded to the same mask cannot accumulate graph pools."""
    from unittest.mock import Mock

    from vllm.v1.rapidserve.worker import SharedModelWorker

    worker = SharedModelWorker.__new__(SharedModelWorker)
    worker.device = torch.device("cuda:0")
    worker.resource_key = ()
    worker.resource_epoch = 0
    worker.cu_fraction = 0
    worker.default_stream = Mock()
    worker.capture_stream = Mock()
    worker.streams = {}
    worker.graphs = {}
    worker.vllm_config = SimpleNamespace(
        additional_config={"rapidserve_role": "decode"}
    )
    worker.model_runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(cudagraph_mode=True),
        decode_query_len=1,
        lora_capture_cases=[],
        capture_model=Mock(),
    )
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "set_stream", Mock())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=20),
    )
    create_stream = Mock(return_value=SimpleNamespace(stream=Mock()))
    monkeypatch.setattr(
        "vllm.v1.rapidserve.worker.current_platform.create_cu_masked_stream",
        create_stream,
    )
    monkeypatch.setattr(
        "vllm.v1.rapidserve.worker.current_platform.graph_pool_handle", Mock()
    )
    monkeypatch.setattr(
        "vllm.v1.rapidserve.worker.ModelCudaGraphManager",
        lambda *_, **__: SimpleNamespace(cudagraph_mode=True),
    )
    worker.set_resource_policy(1, 0.501)
    manager = worker.model_runner.cudagraph_manager
    assert manager.capture_stream is worker.capture_stream
    assert worker.model_runner.main_stream is create_stream.return_value.stream
    worker.set_resource_policy(2, 0.502)
    create_stream.assert_called_once()
    worker.model_runner.capture_model.assert_called_once()
    synchronize.assert_called_once()
    assert worker.resource_epoch == 2
    assert len(worker.graphs) == 2
