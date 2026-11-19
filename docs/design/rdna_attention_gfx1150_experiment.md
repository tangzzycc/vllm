# gfx1150 RDNA attention compatibility experiment

This experimental branch builds the upstream RDNA3.5 attention kernels for
gfx1150 and allows `ROCM_RDNA_ATTN` to select them on Radeon 890M.
Enable it explicitly with `--attention-backend ROCM_RDNA_ATTN --block-size 16`.
Unsupported shapes and features retain the upstream Triton fallback.

The gfx1151-specific 128 KiB LDS occupancy adjustment remains limited to
gfx1151. The experiment does not override the GPU architecture or quantize
weights or KV cache. Compatibility and performance must be verified on the
actual workload. Decoder graphs, multimodal bidirectional masks, and both E4B
head dimensions are included in the local validation.
