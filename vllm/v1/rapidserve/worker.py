# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from typing import Any, cast

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_torch_threads_for_runtime
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.rapidserve.ipc import TensorViews, export_model, import_model
from vllm.v1.rapidserve.resources import partition_cus
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu_worker import Worker


class SharedModelWorker(Worker):
    _weight_export: TensorViews | None = None
    _kv_export: TensorViews | None = None
    resource_epoch: int = 0

    def execute_paired_batch(self, batch, timed=False):
        """Return through WorkerProc's copy thread after local KV writes finish."""
        start = torch.cuda.Event(enable_timing=True) if timed else None
        if start is not None:
            start.record()
        self.execute_model(batch)
        output = self.sample_tokens(None)
        event = torch.cuda.Event(enable_timing=timed)
        event.record()
        return RankOutput(output, start, event)

    def finish_paired_requests(self, batch):
        cast(GPUModelRunner, self.model_runner).finish_requests(batch)

    def import_rank_kv(self, configs, manifests):
        if len(configs) != self.parallel_config.world_size or len(manifests) != len(
            configs
        ):
            raise ValueError("KV IPC manifests must cover every TP rank")
        self.import_kv(configs[self.rank], manifests[self.rank])

    def init_device(self):
        options = self.vllm_config.additional_config
        assert isinstance(options, dict)
        weights = options.get("rapidserve_weights")
        if isinstance(weights, list):
            if len(weights) != self.parallel_config.world_size:
                raise ValueError("Weight IPC manifests must cover every TP rank")
            manifest = weights[self.rank]
            total = current_platform.get_device_total_memory(self.local_rank)
            shared = sum(storage[2] for storage in manifest.storages)
            self.cache_config.gpu_memory_utilization = min(
                self.cache_config.gpu_memory_utilization,
                0.5 * (total - shared) / total,
            )
        super().init_device()
        self.stream_owner = None
        self.default_stream = torch.cuda.Stream(device=self.device)
        self.default_stream.wait_stream(torch.cuda.current_stream(self.device))
        # HIP CU-masked streams are blocking and cannot be used by PyTorch capture.
        # Graph replay uses the selected role stream below.
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.graphs: dict[tuple[int, ...], ModelCudaGraphManager] = {}
        self.streams: dict[tuple[int, ...], Any] = {}
        self.resource_key: tuple[int, ...] = ()
        options = self.vllm_config.additional_config
        assert isinstance(options, dict)
        self.cu_fraction = float(options.get("rapidserve_cu_fraction", 0))
        self.set_resource_policy(0, self.cu_fraction, capture=False)

    def set_resource_policy(self, epoch: int, fraction: float, capture: bool = True):
        """Change streams only after both roles have stopped submitting work."""
        if epoch < self.resource_epoch:
            raise ValueError("Stale resource policy epoch")
        runner = cast(GPUModelRunner, self.model_runner)
        key: tuple[int, ...] = ()
        if fraction:
            total = torch.cuda.get_device_properties(self.device).multi_processor_count
            masks = partition_cus(total, fraction)
            options = self.vllm_config.additional_config
            assert isinstance(options, dict)
            key = tuple(masks[int(options["rapidserve_role"] == "decode")])
        if capture and key == self.resource_key:
            self.cu_fraction = fraction
            self.resource_epoch = epoch
            return epoch
        torch.accelerator.synchronize()
        if (
            capture
            and runner.cudagraph_manager is not None
            and runner.cudagraph_manager.cudagraph_mode
        ):
            self.graphs[self.resource_key] = runner.cudagraph_manager
        owner = self.streams.get(key)
        if fraction and owner is None:
            owner = current_platform.create_cu_masked_stream(list(key), self.device)
            self.streams[key] = owner
        stream = owner.stream if owner is not None else self.default_stream
        torch.cuda.set_stream(stream)
        runner.main_stream = stream
        self.stream_owner = owner
        if (
            capture
            and runner.cudagraph_manager is not None
            and runner.cudagraph_manager.cudagraph_mode
        ):
            manager = self.graphs.get(key)
            if manager is None:
                manager = ModelCudaGraphManager(
                    self.vllm_config,
                    self.device,
                    runner.cudagraph_manager.cudagraph_mode,
                    runner.decode_query_len,
                    lora_capture_cases=runner.lora_capture_cases,
                )
                runner.cudagraph_manager = manager
                manager.capture_stream = self.capture_stream
                manager.pool = current_platform.graph_pool_handle()
                runner.capture_model()
                self.graphs[key] = manager
            runner.cudagraph_manager = manager
        self.cu_fraction = fraction
        self.resource_key = key
        self.resource_epoch = epoch
        return epoch

    def advance_resource_policy(self, epoch: int, fraction: float):
        """Order subsequent batches on a prewarmed policy without a host wait."""
        if epoch < self.resource_epoch:
            raise ValueError("Stale resource policy epoch")
        runner = cast(GPUModelRunner, self.model_runner)
        key: tuple[int, ...] = ()
        if fraction:
            total = torch.cuda.get_device_properties(self.device).multi_processor_count
            masks = partition_cus(total, fraction)
            options = self.vllm_config.additional_config
            assert isinstance(options, dict)
            key = tuple(masks[int(options["rapidserve_role"] == "decode")])
        owner = self.streams.get(key)
        if fraction and owner is None:
            raise ValueError("Resource policy stream was not prewarmed")
        manager = runner.cudagraph_manager
        if manager is not None and manager.cudagraph_mode:
            manager = self.graphs.get(key)
            if manager is None:
                raise ValueError("Resource policy graphs were not prewarmed")
        stream = owner.stream if owner is not None else self.default_stream
        if key != self.resource_key:
            previous = torch.cuda.Event()
            previous.record(runner.main_stream)
            stream.wait_event(previous)
            torch.cuda.set_stream(stream)
            runner.main_stream = stream
            runner.cudagraph_manager = manager
            self.stream_owner = owner
        self.cu_fraction = fraction
        self.resource_key = key
        self.resource_epoch = epoch
        return epoch

    @torch.inference_mode()
    def compile_or_warm_up_model(self):
        runner = cast(GPUModelRunner, self.model_runner)
        if runner.cudagraph_manager is not None:
            runner.cudagraph_manager.capture_stream = self.capture_stream
        result = super().compile_or_warm_up_model()
        runner = cast(GPUModelRunner, self.model_runner)
        assert runner.sampler is not None
        # A fixed KV budget skips profile_run, including its sampler warmup.
        logits = torch.zeros(
            (runner.max_num_reqs, runner.sampler.sampling_states.vocab_size),
            dtype=self.model_config.dtype,
            device=self.device,
        )
        indices = torch.arange(runner.max_num_reqs, device=self.device)
        temperature = torch.zeros(runner.max_num_reqs, device=self.device)
        seeds = torch.zeros_like(indices)
        for apply_temperature in (False, True):
            gumbel_sample(
                logits,
                indices,
                temperature,
                seeds,
                seeds,
                apply_temperature=apply_temperature,
                is_drafting=False,
                use_fp64=runner.sampler.use_fp64_gumbel,
            )
        torch.accelerator.synchronize()
        return result

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        options = self.vllm_config.additional_config
        assert isinstance(options, dict)
        manifest = options.pop("rapidserve_weights", None)
        if manifest is None:
            super().load_model(load_dummy_weights=load_dummy_weights)
            return
        if isinstance(manifest, list):
            if len(manifest) != self.parallel_config.world_size:
                raise ValueError("Weight IPC manifests must cover every TP rank")
            manifest = manifest[self.rank]
        with set_current_vllm_config(self.vllm_config):
            runner = cast(GPUModelRunner, self.model_runner)
            with runner.jit_warmup_registry.activate():
                runner.model = import_model(self.vllm_config, manifest)
            runner.model_memory_usage = 0
            runner.initialize_model_components()
        set_torch_threads_for_runtime()

    def export_weights(self) -> TensorViews:
        torch.accelerator.synchronize()
        self._weight_export = export_model(self.model_runner.model)
        return self._weight_export

    def export_kv(self) -> TensorViews:
        torch.accelerator.synchronize()
        layers = self.vllm_config.compilation_config.static_forward_context
        self._kv_export = TensorViews.export(
            {
                name: layer.kv_cache
                for name, layer in layers.items()
                if hasattr(layer, "kv_cache") and layer.kv_cache.numel()
            }
        )

        return self._kv_export

    def import_kv(self, config, manifest: TensorViews) -> None:
        if config.kv_cache_layout is not None:
            record_kv_cache_layout(self.cache_config, config.kv_cache_layout)
        self.cache_config.num_gpu_blocks = config.num_blocks
        self.cache_config.hash_block_size = config.hash_block_size
        self.cache_config.cache_hit_alignment_tokens = config.cache_hit_alignment_tokens
        cast(GPUModelRunner, self.model_runner).initialize_kv_cache(
            config, shared_kv_caches=manifest.import_tensors(self.device)
        )

    def shutdown(self) -> None:
        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        tensors: list[torch.Tensor | None] = []
        if model is not None:
            for module in model.modules():
                tensors.extend(module._parameters.values())
                tensors.extend(module._buffers.values())
                tensors.extend(
                    value
                    for value in vars(module).values()
                    if isinstance(value, torch.Tensor)
                )
        super().shutdown()
        # Compiled functions can retain Tensor objects after their module is freed.
        # Detach their storage after GPU completion and before the IPC owner exits.
        for tensor in tensors:
            if tensor is not None:
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        tensors.clear()
        if hasattr(self, "graphs"):
            self.graphs.clear()
        wrappers = list(CUDAGraphWrapper._all_instances) + list(
            BreakableCUDAGraphWrapper._all_instances
        )
        for wrapper in wrappers:
            if wrapper.vllm_config is self.vllm_config:
                wrapper.clear_graphs()
        if model is not None:
            from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

            # Compiled callables can retain modules beyond the runner's lifetime.
            for module in model.modules():
                if isinstance(module, TorchCompileWithNoGuardsWrapper):
                    module.cleanup()
                    if hasattr(module, "aot_compiled_fn"):
                        module.aot_compiled_fn = None
                    module._compiled_callable = None
                if isinstance(module, MoERunner) and hasattr(
                    module.router, "e_score_correction_bias"
                ):
                    module.router.e_score_correction_bias = None
                module._parameters.clear()
                module._buffers.clear()
                for name, value in list(vars(module).items()):
                    if isinstance(value, torch.Tensor):
                        delattr(module, name)
            del module, model
        if not hasattr(self, "default_stream"):
            return
        torch.cuda.set_stream(self.default_stream)
        for owner in self.streams.values():
            owner.close()
        self.streams.clear()
        self.stream_owner = None
        self._weight_export = None
        self._kv_export = None
        gc.collect()
        torch.accelerator.empty_cache()


class RankOutput(AsyncModelRunnerOutput):
    def __init__(self, output, start, event):
        self.output = output
        self.start = start
        self.event = event

    def get_output(self):
        self.event.synchronize()
        output = self.output.get_output() if self.output is not None else None
        duration = (
            self.start.elapsed_time(self.event) if self.start is not None else 0.0
        )
        return output, duration
