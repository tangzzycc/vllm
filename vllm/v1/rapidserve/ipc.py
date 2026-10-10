# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.multiprocessing.reductions import StorageWeakRef, shared_cache

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype


@dataclass
class TensorViews:
    """One IPC export per storage, with the original named views."""

    storages: list[tuple[Any, ...]]
    views: dict[str, tuple[int, torch.dtype, tuple[int, ...], tuple[int, ...], int]]

    device_uuid: str
    host_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    _owners: list[torch.Tensor] = field(default_factory=list, repr=False)

    def __getstate__(self):
        return {
            "storages": self.storages,
            "views": self.views,
            "device_uuid": self.device_uuid,
            "host_tensors": self.host_tensors,
            "_owners": [],
        }

    @classmethod
    def export(cls, tensors: dict[str, torch.Tensor]) -> "TensorViews":
        storages: list[tuple[Any, ...]] = []
        views = {}
        indices = {}
        owners = []
        for name, tensor in tensors.items():
            if not tensor.is_cuda or tensor.requires_grad:
                raise ValueError("IPC requires accelerator tensors without gradients")
            storage = tensor.untyped_storage()
            key = storage._cdata
            if key not in indices:
                flat = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                flat.set_(storage, 0, (storage.nbytes(),), (1,))
                storage_ref = flat._typed_storage()
                args = storage._share_cuda_()
                shared_cache[args[1]] = StorageWeakRef(storage_ref)
                indices[key] = len(storages)
                storages.append(args)
                owners.append(flat)
            views[name] = (
                indices[key],
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )
        device_uuid = str(
            torch.cuda.get_device_properties(next(iter(tensors.values())).device).uuid
        )
        return cls(storages, views, device_uuid, _owners=owners)

    def import_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        if str(torch.cuda.get_device_properties(device).uuid) != self.device_uuid:
            raise ValueError("IPC storage belongs to a different physical GPU")
        imported = []
        for stored_args in self.storages:
            args = list(stored_args)
            args[0] = device.index or 0
            storage = torch.UntypedStorage._new_shared_cuda(*args)
            tensor = torch.empty(0, dtype=torch.uint8, device=device)
            tensor.set_(storage, 0, (storage.nbytes(),), (1,))
            imported.append(tensor)
        result = {name: tensor.clone() for name, tensor in self.host_tensors.items()}
        for name, (index, dtype, shape, stride, offset) in self.views.items():
            tensor = torch.empty(0, dtype=dtype, device=device)
            tensor.set_(imported[index].untyped_storage(), offset, shape, stride)
            result[name] = tensor
        return result


def export_model(model: torch.nn.Module) -> TensorViews:
    tensors = {
        "parameter:" + name: param.detach()
        for name, param in model.named_parameters(remove_duplicate=False)
    }
    tensors.update(
        {
            "buffer:" + name: buf.detach()
            for name, buf in model.named_buffers(remove_duplicate=False)
        }
    )
    for path, module in model.named_modules():
        for name, value in vars(module).items():
            if isinstance(value, torch.Tensor) and name != "kv_cache":
                tensors[f"attribute:{path}.{name}"] = value.detach()
    manifest = TensorViews.export(
        {name: tensor for name, tensor in tensors.items() if tensor.is_cuda}
    )
    manifest.host_tensors = {
        name: tensor.clone()
        for name, tensor in tensors.items()
        if tensor.device.type == "cpu"
    }
    return manifest


def validate_shared_model(config: VllmConfig) -> None:
    model = config.model_config
    parallel = config.parallel_config
    if (
        not config.use_v2_model_runner
        or parallel.world_size != 1
        or parallel.data_parallel_size != 1
        or parallel.use_ubatching
        or model.is_multimodal_model
        or model.is_moe
        or model.is_hybrid
        or model.is_attention_free
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
        or config.cache_config.kv_cache_dtype_skip_layers
        or config.weight_transfer_config is not None
        or config.watermark_config is not None
        or model.enable_trace_replay
        or model.return_sampling_mask
        or model.is_diffusion
    ):
        raise ValueError(
            "Shared MRV2 storage requires single-GPU resident dense text models"
        )


@torch.inference_mode()
def import_model(config: VllmConfig, manifest: TensorViews) -> torch.nn.Module:
    """Construct a local model and attach shared weights without a second load."""
    validate_shared_model(config)
    device = config.device_config.device
    with set_default_torch_dtype(config.model_config.dtype), torch.device("meta"):
        model = initialize_model(config)
    tensors = manifest.import_tensors(device)
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases = {}
    for key, tensor in tensors.items():
        kind, path = key.split(":", 1)
        parent, _, name = path.rpartition(".")
        module = modules[parent]
        if kind == "parameter":
            alias = (
                tensor.untyped_storage()._cdata,
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            if alias not in aliases:
                aliases[alias] = torch.nn.Parameter(tensor, requires_grad=False)
            module._parameters[name] = aliases[alias]
        elif kind == "buffer":
            # Buffers such as RoPE caches may be replaced during forward.
            module._buffers[name] = tensor.clone()
        else:
            setattr(module, name, tensor.clone())
    missing = [name for name, tensor in model.named_parameters() if tensor.is_meta]
    missing.extend(name for name, tensor in model.named_buffers() if tensor.is_meta)
    for path, module in model.named_modules():
        missing.extend(
            f"{path}.{name}"
            for name, tensor in vars(module).items()
            if isinstance(tensor, torch.Tensor)
            and tensor.is_meta
            and name != "kv_cache"
        )
    if missing:
        raise ValueError(f"IPC model has uninitialized tensors: {missing}")
    return model.eval()
