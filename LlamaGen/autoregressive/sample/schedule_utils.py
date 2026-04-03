import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
EVOAR_DIR = ROOT_DIR / "EvoAR-v1"
if str(EVOAR_DIR) not in sys.path:
    sys.path.insert(0, str(EVOAR_DIR))

from autoregressive.train.mask_builder import build_grouped_attention_mask, canonicalize_schedule


def load_schedule_from_checkpoint(ckpt_path, schedule_source="pareto_front", schedule_index=0, schedule=None, map_location="cpu"):
    if schedule is not None:
        return canonicalize_schedule(torch.as_tensor(schedule, dtype=torch.long))

    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if schedule_source not in checkpoint:
        raise ValueError(f"checkpoint does not contain schedule source: {schedule_source}")
    candidates = checkpoint[schedule_source]
    if not candidates:
        raise ValueError(f"checkpoint schedule source is empty: {schedule_source}")
    if schedule_index < 0 or schedule_index >= len(candidates):
        raise IndexError(f"schedule_index {schedule_index} out of range for {schedule_source} with size {len(candidates)}")
    item = candidates[schedule_index]
    raw_schedule = item["schedule"] if isinstance(item, dict) else item
    return canonicalize_schedule(torch.as_tensor(raw_schedule, dtype=torch.long))


def build_inference_mask(cond, schedule):
    batch_size = cond.shape[0]
    if cond.ndim == 1:
        prefix_valid_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=cond.device)
    elif cond.ndim == 3:
        prefix_valid_mask = torch.ones((batch_size, cond.shape[1]), dtype=torch.bool, device=cond.device)
    else:
        raise ValueError(f"unsupported cond shape for inference mask: {tuple(cond.shape)}")
    schedule_steps = schedule.to(cond.device).unsqueeze(0).expand(batch_size, -1)
    return build_grouped_attention_mask(prefix_valid_mask, schedule_steps)
