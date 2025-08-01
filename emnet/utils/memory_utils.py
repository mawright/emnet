import torch
import psutil
import gc
from typing import Optional, Union
from lightning.fabric import Fabric


def collect_memory_stats(
    device: Optional[Union[int, torch.device]] = None,
    include_reserved: bool = True,
    force_gc: bool = False,
) -> dict[str, float]:
    if force_gc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        return {}

    stats = {}

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    assert isinstance(device, torch.device)
    device_id = device.index
    prefix = f"cuda_memory/{device_id}/"
    MB = 1024 ** 2

    allocated = torch.cuda.memory_allocated(device)
    cached = torch.cuda.memory_reserved(device)

    stats.update({
        prefix + "allocated_mb": allocated / MB,
        prefix + "cached_mb": cached / MB,
    })
