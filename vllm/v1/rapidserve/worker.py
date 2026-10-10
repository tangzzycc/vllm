# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

from vllm.config import set_current_vllm_config
from vllm.utils.torch_utils import set_torch_threads_for_runtime
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.rapidserve.ipc import TensorViews, export_model, import_model
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


class SharedModelWorker(Worker):
    _weight_export: TensorViews | None = None
    _kv_export: TensorViews | None = None

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        options = self.vllm_config.additional_config
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
        super().shutdown()
        self._weight_export = None
        self._kv_export = None
