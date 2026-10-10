# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import torch


class CuMaskedStream:
    """Own a HIP stream and its non-owning PyTorch wrapper."""

    def __init__(self, mask: list[int], device: torch.device):
        if not mask or not any(mask) or any(word < 0 or word >= 2**32 for word in mask):
            raise ValueError("CU mask must contain nonzero uint32 words")
        if torch.cuda.current_device() != device.index:
            raise ValueError("Set the current HIP device before creating a CU stream")
        self.runtime = ctypes.CDLL("libamdhip64.so")
        self.runtime.hipExtStreamCreateWithCUMask.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.runtime.hipExtStreamGetCUMask.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.runtime.hipStreamDestroy.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        words = (ctypes.c_uint32 * len(mask))(*mask)
        status = self.runtime.hipExtStreamCreateWithCUMask(
            ctypes.byref(self.handle), len(mask), words
        )
        if status:
            raise RuntimeError(f"HIP CU stream creation failed: {status}")
        try:
            actual = (ctypes.c_uint32 * len(mask))()
            status = self.runtime.hipExtStreamGetCUMask(self.handle, len(mask), actual)
            if status or list(actual) != mask:
                raise RuntimeError(
                    f"HIP CU mask readback mismatch: {status}, {list(actual)}"
                )
            self.mask = tuple(actual)
            assert self.handle.value is not None
            self.stream = torch.cuda.ExternalStream(self.handle.value, device=device)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.handle.value is not None:
            status = self.runtime.hipStreamDestroy(self.handle)
            self.handle = ctypes.c_void_p()
            if status:
                raise RuntimeError(f"HIP CU stream destruction failed: {status}")
