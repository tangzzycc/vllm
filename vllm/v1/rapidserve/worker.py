# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import torch

from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_torch_threads_for_runtime
from vllm.v1.attention.backends.utils import record_kv_cache_layout
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

    def init_device(self):
        super().init_device()
        self.stream_owner = None
        self.default_stream = torch.cuda.Stream(device=self.device)
        self.default_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.graphs: dict[float, ModelCudaGraphManager] = {}
        self.streams: dict[float, Any] = {}
        options = self.vllm_config.additional_config
        assert isinstance(options, dict)
        self.cu_fraction = float(options.get("rapidserve_cu_fraction", 0))
        self.set_resource_policy(0, self.cu_fraction, capture=False)

    def set_resource_policy(self, epoch: int, fraction: float, capture: bool = True):
        """Change streams only after both roles have stopped submitting work."""
        if epoch < self.resource_epoch:
            raise ValueError("Stale resource policy epoch")
        runner = cast(GPUModelRunner, self.model_runner)
        torch.accelerator.synchronize()
        if (
            capture
            and runner.cudagraph_manager is not None
            and runner.cudagraph_manager.cudagraph_mode
        ):
            self.graphs[self.cu_fraction] = runner.cudagraph_manager
        owner = self.streams.get(fraction)
        if fraction and owner is None:
            total = torch.cuda.get_device_properties(self.device).multi_processor_count
            masks = partition_cus(total, fraction)
            options = self.vllm_config.additional_config
            assert isinstance(options, dict)
            role = options["rapidserve_role"]
            mask = masks[int(role == "decode")]
            owner = current_platform.create_cu_masked_stream(mask, self.device)
            self.streams[fraction] = owner
        stream = owner.stream if owner is not None else self.default_stream
        torch.cuda.set_stream(stream)
        runner.main_stream = stream
        self.stream_owner = owner
        if (
            capture
            and runner.cudagraph_manager is not None
            and runner.cudagraph_manager.cudagraph_mode
        ):
            manager = self.graphs.get(fraction)
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
                self.graphs[fraction] = manager
            runner.cudagraph_manager = manager
        self.cu_fraction = fraction
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
        manifest = (
            options.get("rapidserve_weights") if isinstance(options, dict) else None
        )
        if manifest is None:
            super().load_model(load_dummy_weights=load_dummy_weights)
            return
        with set_current_vllm_config(self.vllm_config):
            runner = cast(GPUModelRunner, self.model_runner)
            runner.model = import_model(self.vllm_config, manifest)
            runner.model_memory_usage = 0
            runner.initialize_model_components()
        set_torch_threads_for_runtime()

    def export_weights(self) -> TensorViews:
        self._weight_export = export_model(self.model_runner.model)
        return self._weight_export

    def export_kv(self) -> TensorViews:
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
        cast(GPUModelRunner, self.model_runner).initialize_kv_cache(
            config, shared_kv_caches=manifest.import_tensors(self.device)
        )

    def shutdown(self) -> None:
        if hasattr(self, "graphs"):
            self.graphs.clear()
        super().shutdown()
        if not hasattr(self, "default_stream"):
            return
        torch.cuda.set_stream(self.default_stream)
        for owner in self.streams.values():
            owner.close()
        self.streams.clear()
        self.stream_owner = None
        self._weight_export = None
        self._kv_export = None
