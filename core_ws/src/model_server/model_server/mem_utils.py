"""Memory hygiene for the model servers.

Loading a multi-GB torch checkpoint streams it through the CPU heap before the
weights land on the GPU. Python frees those tensors immediately, but glibc keeps
the freed arenas, so the process's RSS stays at the load-time high-water mark for
the rest of its life — sam_server sat at ~7.0 GB when its weights are only 3.4 GB,
and graspgen_server does the same with its generator + discriminator checkpoints.

On this 30 GB / 2 GB-swap box that phantom memory is the difference between the
sim + models fitting alongside the desktop and the host thrashing into a hard
freeze (it caused two, 2026-07-13). malloc_trim() actually hands the arenas back
to the OS: sam_server drops 6960 MB -> 1021 MB.

Call release_load_memory() once, right after the model is built.
"""
import ctypes
import gc

import torch


def release_load_memory():
    """Return checkpoint-loading heap to the OS (see module docstring)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):        # non-glibc: nothing to trim
        pass
