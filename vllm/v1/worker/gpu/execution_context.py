# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

import torch

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.workspace import use_workspace_lane


@torch.inference_mode()
def create_execution_context(
    owner: GPUModelRunner, workspace_lane: int
) -> GPUModelRunner:
    """Create runner-local execution state over an owner's model and KV storage.

    The owner must finish initialization before this call. The caller provides
    an independent workspace lane and orders execution and storage lifetime.
    This context must not shut down the shared model or global worker state.

    Args:
        owner: Initialized runner that owns the model and physical KV allocation.
        workspace_lane: Workspace lane reserved for the new runner.

    Returns:
        An eager runner with independent request, input, sampling and KV metadata.

    """
    config = owner.vllm_config
    model = config.model_config
    parallel = config.parallel_config
    if (
        not model.enforce_eager
        or config.compilation_config.mode != CompilationMode.NONE
        or parallel.world_size != 1
        or parallel.data_parallel_size != 1
        or parallel.use_ubatching
    ):
        raise ValueError("Shared execution contexts require single-GPU eager execution")
    if (
        type(owner.model_state) is not DefaultModelState
        or model.is_multimodal_model
        or model.is_moe
        or model.quantization is not None
        or model.runner_type != "generate"
        or model.enable_sleep_mode
        or model.logits_processors
        or config.speculative_config is not None
        or config.lora_config is not None
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
        or config.aux_output_config.enabled
        or config.attention_config.hisparse_config is not None
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
    ):
        raise ValueError(
            "Shared execution contexts support resident dense text models "
            "with the default model state and sampler"
        )
    if workspace_lane <= 0:
        raise ValueError("Reserve a workspace lane distinct from the owner's lane 0")
    context_config = copy(config)
    context_config.compilation_config = copy(config.compilation_config)
    with set_current_vllm_config(context_config), use_workspace_lane(workspace_lane):
        runner = GPUModelRunner(context_config, owner.device)
        runner.model = owner.model
        runner.model_memory_usage = 0
        runner.initialize_model_components()
        runner.initialize_kv_cache(
            owner.kv_cache_config, shared_kv_caches=owner.kv_caches
        )
    return runner
