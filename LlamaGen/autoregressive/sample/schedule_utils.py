import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
EVOAR_DIR = ROOT_DIR / "EvoAR-v1"
if str(EVOAR_DIR) not in sys.path:
    sys.path.insert(0, str(EVOAR_DIR))

from autoregressive.train.mask_builder import build_grouped_attention_mask, canonicalize_schedule


def schedule_to_groups(schedule):
    schedule = canonicalize_schedule(torch.as_tensor(schedule, dtype=torch.long))
    num_groups = int(schedule.max().item()) + 1 if schedule.numel() > 0 else 0
    return [torch.nonzero(schedule == step, as_tuple=False).flatten() for step in range(num_groups)]


def build_grouped_full_sequence(schedule, generated=None, device=None):
    schedule = canonicalize_schedule(torch.as_tensor(schedule, dtype=torch.long, device=device))
    code_len = int(schedule.numel())
    if generated is None:
        generated = torch.zeros(code_len, dtype=torch.bool, device=schedule.device)
    else:
        generated = torch.as_tensor(generated, dtype=torch.bool, device=schedule.device)
    if generated.ndim != 1 or generated.numel() != code_len:
        raise ValueError("generated mask must be 1D with same length as schedule")
    return schedule, generated


def build_grouped_step_mask(prefix_valid_mask, schedule, generated, current_positions):
    schedule, generated = build_grouped_full_sequence(schedule, generated, device=prefix_valid_mask.device)
    current_positions = torch.as_tensor(current_positions, dtype=torch.long, device=schedule.device)
    if current_positions.ndim != 1:
        raise ValueError("current_positions must be 1D")

    batch_size = prefix_valid_mask.shape[0]
    prefix_len = prefix_valid_mask.shape[1]
    code_len = schedule.numel()
    total_len = prefix_len + code_len
    mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=schedule.device)

    if prefix_len > 0:
        prefix_causal = torch.tril(torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=schedule.device))
        prefix_prefix = prefix_causal.unsqueeze(0) & prefix_valid_mask[:, None, :]
        mask[:, :prefix_len, :prefix_len] = prefix_prefix
        mask[:, prefix_len:, :prefix_len] = prefix_valid_mask[:, None, :].expand(batch_size, code_len, prefix_len)

    earlier_or_current = generated.clone()
    earlier_or_current[current_positions] = True
    latent_mask = earlier_or_current.unsqueeze(0).expand(code_len, -1)
    latent_mask[current_positions[:, None], current_positions[None, :]] = False
    latent_mask[current_positions, current_positions] = True
    mask[:, prefix_len:, prefix_len:] = latent_mask.unsqueeze(0).expand(batch_size, -1, -1)
    mask = mask | torch.eye(total_len, dtype=torch.bool, device=schedule.device).unsqueeze(0)
    return mask.unsqueeze(1)


def build_grouped_block_mask(prefix_valid_mask, schedule, generated, current_positions):
    current_positions = torch.as_tensor(current_positions, dtype=torch.long, device=prefix_valid_mask.device)
    full_mask = build_grouped_step_mask(prefix_valid_mask, schedule, generated, current_positions)
    prefix_len = prefix_valid_mask.shape[1]
    query_rows = prefix_len + current_positions
    return full_mask[:, :, query_rows, :]


def build_grouped_compact_layout(prefix_valid_mask, generated, current_positions):
    current_positions = torch.as_tensor(current_positions, dtype=torch.long, device=prefix_valid_mask.device)
    generated = torch.as_tensor(generated, dtype=torch.bool, device=prefix_valid_mask.device)
    earlier_positions = torch.nonzero(generated, as_tuple=False).flatten()
    key_positions = torch.cat([earlier_positions, current_positions], dim=0)
    compact_len = prefix_valid_mask.shape[1] + key_positions.numel()
    return earlier_positions, key_positions, compact_len


def build_grouped_compact_mask(prefix_valid_mask, generated, current_positions):
    current_positions = torch.as_tensor(current_positions, dtype=torch.long, device=prefix_valid_mask.device)
    earlier_positions, key_positions, compact_len = build_grouped_compact_layout(prefix_valid_mask, generated, current_positions)
    batch_size = prefix_valid_mask.shape[0]
    prefix_len = prefix_valid_mask.shape[1]
    current_len = current_positions.numel()
    mask = torch.zeros((batch_size, current_len, compact_len), dtype=torch.bool, device=prefix_valid_mask.device)

    if prefix_len > 0:
        mask[:, :, :prefix_len] = prefix_valid_mask[:, None, :].expand(batch_size, current_len, prefix_len)

    if earlier_positions.numel() > 0:
        earlier_start = prefix_len
        earlier_end = earlier_start + earlier_positions.numel()
        mask[:, :, earlier_start:earlier_end] = True

    current_start = prefix_len + earlier_positions.numel()
    if current_len > 0:
        eye = torch.eye(current_len, dtype=torch.bool, device=prefix_valid_mask.device).unsqueeze(0)
        mask[:, :, current_start:current_start + current_len] = eye.expand(batch_size, -1, -1)

    return mask.unsqueeze(1), earlier_positions, key_positions


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


def _format_positions(positions, max_items=16):
    positions = torch.as_tensor(positions, dtype=torch.long).flatten().tolist()
    if not positions:
        return "[]"
    if len(positions) <= max_items:
        return "[" + ",".join(str(int(pos)) for pos in positions) + "]"
    head = ",".join(str(int(pos)) for pos in positions[:max_items])
    return f"[{head},...](count={len(positions)})"


def _format_sizes(sizes, max_items=12):
    if len(sizes) <= max_items:
        return str(sizes)
    head = ", ".join(str(int(size)) for size in sizes[:max_items])
    return f"[{head}, ...] (total_steps={len(sizes)})"


def build_semantic_check_lines(prefix_valid_mask, schedule, tag, mask_mode="full", note=None):
    prefix_valid_mask = torch.as_tensor(prefix_valid_mask, dtype=torch.bool)
    if prefix_valid_mask.ndim == 1:
        prefix_valid_mask = prefix_valid_mask.unsqueeze(0)
    prefix_valid_mask = prefix_valid_mask[:1]
    schedule = canonicalize_schedule(torch.as_tensor(schedule, dtype=torch.long, device=prefix_valid_mask.device))
    grouped_positions = schedule_to_groups(schedule)
    step_sizes = [int(positions.numel()) for positions in grouped_positions]
    prefix_len = int(prefix_valid_mask.shape[1])
    valid_prefix = torch.nonzero(prefix_valid_mask[0], as_tuple=False).flatten()

    lines = [
        f"[semantic-check][{tag}] prefix_len={prefix_len} code_len={int(schedule.numel())} generation_steps={len(grouped_positions)} step_sizes={_format_sizes(step_sizes)} mask_mode={mask_mode}"
    ]
    if note is not None:
        lines.append(f"[semantic-check][{tag}] note={note}")

    if not grouped_positions:
        return lines

    inspected_steps = []
    for candidate in (0, 1, len(grouped_positions) - 1):
        if 0 <= candidate < len(grouped_positions) and candidate not in inspected_steps:
            inspected_steps.append(candidate)

    generated = torch.zeros(schedule.shape[0], dtype=torch.bool, device=schedule.device)
    for step, positions in enumerate(grouped_positions):
        if step in inspected_steps:
            earlier_positions = torch.nonzero(generated, as_tuple=False).flatten()
            future_mask = (~generated).clone()
            future_mask[positions] = False
            future_positions = torch.nonzero(future_mask, as_tuple=False).flatten()

            if mask_mode == "full":
                full_mask = build_grouped_step_mask(prefix_valid_mask, schedule, generated, positions)[0, 0]
                query_rows = prefix_len + positions
                query_view = full_mask[query_rows]
                prefix_visible = True if valid_prefix.numel() == 0 else bool(query_view[:, valid_prefix].all().item())
                earlier_visible = True if earlier_positions.numel() == 0 else bool(query_view[:, prefix_len + earlier_positions].all().item())
                current_view = query_view[:, prefix_len + positions]
                if positions.numel() > 0:
                    eye = torch.eye(positions.numel(), dtype=torch.bool, device=current_view.device)
                    self_visible = bool(current_view[eye].all().item())
                    same_group_cross_visible = bool((current_view & ~eye).any().item())
                else:
                    self_visible = True
                    same_group_cross_visible = False
                future_visible = False if future_positions.numel() == 0 else bool(query_view[:, prefix_len + future_positions].any().item())
                lines.append(
                    f"[semantic-check][{tag}] step={step} current={_format_positions(positions)} earlier={_format_positions(earlier_positions)} future_count={int(future_positions.numel())} checks: prefix_visible={prefix_visible} earlier_visible={earlier_visible} self_visible={self_visible} same_group_cross_visible={same_group_cross_visible} future_visible={future_visible}"
                )
            elif mask_mode == "compact":
                compact_mask, earlier_positions_compact, key_positions = build_grouped_compact_mask(prefix_valid_mask, generated, positions)
                query_view = compact_mask[0, 0]
                prefix_visible = True if valid_prefix.numel() == 0 else bool(query_view[:, valid_prefix].all().item())
                earlier_len = int(earlier_positions_compact.numel())
                earlier_visible = True if earlier_len == 0 else bool(query_view[:, prefix_len:prefix_len + earlier_len].all().item())
                current_view = query_view[:, prefix_len + earlier_len:]
                if positions.numel() > 0:
                    eye = torch.eye(positions.numel(), dtype=torch.bool, device=current_view.device)
                    self_visible = bool(current_view[eye].all().item())
                    same_group_cross_visible = bool((current_view & ~eye).any().item())
                else:
                    self_visible = True
                    same_group_cross_visible = False
                lines.append(
                    f"[semantic-check][{tag}] step={step} current={_format_positions(positions)} earlier={_format_positions(earlier_positions)} future_count={int(future_positions.numel())} compact_len={prefix_len + int(key_positions.numel())} checks: prefix_visible={prefix_visible} earlier_visible={earlier_visible} self_visible={self_visible} same_group_cross_visible={same_group_cross_visible} future_in_graph=False"
                )
            else:
                raise ValueError(f"unsupported mask_mode: {mask_mode}")
        generated[positions] = True

    return lines


def analyze_schedule_realizability(schedule):
    schedule = canonicalize_schedule(torch.as_tensor(schedule, dtype=torch.long))
    code_len = int(schedule.numel())
    positions_with_future_dependencies = 0
    future_dependency_edges = 0

    for pos in range(code_len):
        future_dependencies = int((schedule[pos + 1 :] < schedule[pos]).sum().item())
        if future_dependencies > 0:
            positions_with_future_dependencies += 1
            future_dependency_edges += future_dependencies

    return {
        "code_len": code_len,
        "num_groups": int(schedule.max().item()) + 1 if code_len > 0 else 0,
        "positions_with_future_dependencies": positions_with_future_dependencies,
        "future_dependency_edges": future_dependency_edges,
    }


def build_inference_mask(cond, schedule):
    batch_size = cond.shape[0]
    if cond.ndim == 1:
        prefix_valid_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=cond.device)
    elif cond.ndim == 3:
        prefix_valid_mask = torch.ones((batch_size, cond.shape[1]), dtype=torch.bool, device=cond.device)
    else:
        raise ValueError(f"unsupported cond shape for inference mask: {tuple(cond.shape)}")

    schedule = canonicalize_schedule(torch.as_tensor(schedule, dtype=torch.long, device=cond.device))
    schedule_steps = schedule.unsqueeze(0).expand(batch_size, -1)
    inference_mask = build_grouped_attention_mask(prefix_valid_mask, schedule_steps)

    prefix_len = prefix_valid_mask.shape[1]
    code_len = schedule_steps.shape[1]
    latent_causal = torch.tril(torch.ones((code_len, code_len), dtype=torch.bool, device=cond.device))
    inference_mask[:, :, prefix_len:, prefix_len:] = (
        inference_mask[:, :, prefix_len:, prefix_len:] & latent_causal.unsqueeze(0).unsqueeze(0)
    )
    return inference_mask
