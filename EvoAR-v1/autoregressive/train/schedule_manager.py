import math
import random

import torch
import torch.distributed as dist

from autoregressive.train.mask_builder import canonicalize_schedule


class ScheduleManager:
    def __init__(
        self,
        code_len,
        evolve_every,
        population_size,
        mutation_prob,
        max_groups=None,
        device=None,
        crossover_prob=0.5,
        mutation_weights=None,
        shift_radius=4,
        block_max_size=8,
        split_prob=0.15,
        trend_weight=0.25,
        final_loss_weight=0.75,
    ):
        self.code_len = int(code_len)
        self.evolve_every = max(int(evolve_every), 0)
        self.population_size = max(int(population_size), 1)
        self.mutation_prob = float(mutation_prob)
        self.max_groups = int(max_groups) if max_groups is not None else self.code_len
        self.device = device
        self.crossover_prob = float(crossover_prob)
        self.shift_radius = max(int(shift_radius), 1)
        self.block_max_size = max(int(block_max_size), 1)
        self.split_prob = float(split_prob)
        self.trend_weight = float(trend_weight)
        self.final_loss_weight = float(final_loss_weight)
        self.mutation_weights = mutation_weights or {
            "merge_to_nearby_step": 0.30,
            "local_shift_step": 0.25,
            "block_move": 0.20,
            "local_swap_order": 0.15,
            "split_group": 0.10,
        }

        total_weight = sum(self.mutation_weights.values())
        self.mutation_weights = {key: value / total_weight for key, value in self.mutation_weights.items()}
        self._apply_split_prob()

        self.population = []
        self.archive = []
        self.pending_records = {}
        self.schedule_stats = {}
        self._initialize_population()

    def _initialize_population(self):
        base = self.base_schedule(device=self.device)
        self.population = [base.clone()]
        while len(self.population) < self.population_size:
            candidate = base.clone()
            perturb_steps = random.randint(1, 3)
            for _ in range(perturb_steps):
                candidate = self.apply_mutation(candidate)
            self.population.append(candidate)

    def _apply_split_prob(self):
        current = self.mutation_weights.copy()
        split_original = current.get("split_group", 0.0)
        other_sum = max(1e-8, 1.0 - split_original)
        target_split = min(max(self.split_prob, 0.0), 0.95)
        target_other = 1.0 - target_split
        updated = {}
        for key, value in current.items():
            if key == "split_group":
                updated[key] = target_split
            else:
                updated[key] = value / other_sum * target_other
        self.mutation_weights = updated

    def base_schedule(self, device=None):
        device = device or self.device
        return torch.arange(self.code_len, device=device, dtype=torch.long)

    def sample(self, batch_size, step, device=None):
        device = device or self.device
        schedules = []
        for _ in range(batch_size):
            selected = self.population[random.randrange(len(self.population))]
            schedules.append(selected.clone())
        return torch.stack(schedules, dim=0).to(device)

    def record(self, schedule_steps, sample_loss, latency_proxy):
        for idx in range(schedule_steps.shape[0]):
            schedule = canonicalize_schedule(schedule_steps[idx].detach().cpu().long())
            key = self._schedule_key(schedule)
            record = self.pending_records.get(key)
            if record is None:
                record = {
                    "schedule": schedule,
                    "loss_sum": 0.0,
                    "latency_sum": 0.0,
                    "count": 0,
                }
                self.pending_records[key] = record
            record["loss_sum"] += float(sample_loss[idx].detach().cpu().item())
            record["latency_sum"] += float(latency_proxy[idx].detach().cpu().item())
            record["count"] += 1

    def flush_pending_records(self):
        flushed = list(self.pending_records.values())
        self.pending_records = {}
        return flushed

    def ingest_records(self, records):
        for record in records:
            schedule = canonicalize_schedule(record["schedule"].detach().cpu().long())
            key = self._schedule_key(schedule)
            stats = self.schedule_stats.get(key)
            if stats is None:
                stats = {
                    "schedule": schedule,
                    "loss_history": [],
                    "latency_history": [],
                    "count": 0,
                }
                self.schedule_stats[key] = stats
            count = max(int(record.get("count", 0)), 1)
            avg_loss = float(record.get("loss", record["loss_sum"] / count))
            avg_latency = float(record.get("latency", record["latency_sum"] / count))
            stats["loss_history"].append(avg_loss)
            stats["latency_history"].append(avg_latency)
            stats["count"] += count

    def should_evolve(self, step):
        return self.evolve_every > 0 and step > 0 and step % self.evolve_every == 0

    def evolve_if_needed(self, step):
        if not self.should_evolve(step) or not self._has_cycle_stats():
            return False

        evaluated = self._build_evaluated_population()
        if not evaluated:
            self.pending_records = {}
            self._reset_cycle_stats()
            return False

        parent_pool = self._limit_candidates(evaluated, max(self.population_size, self.population_size * 2))
        offspring = self._generate_offspring(parent_pool)
        combined = self._deduplicate_candidates(evaluated + offspring)
        combined = self._limit_candidates(combined, max(self.population_size, self.population_size * 4))
        selected = self._nsga2_select(combined, self.population_size)
        if not selected:
            self.pending_records = {}
            self._reset_cycle_stats()
            return False

        self.population = [item["schedule"].to(self.device) for item in selected]
        self.archive = self._build_archive(combined)
        self.pending_records = {}
        self._reset_cycle_stats()
        return True

    def _reset_cycle_stats(self):
        for stats in self.schedule_stats.values():
            stats["loss_history"] = []
            stats["latency_history"] = []
            stats["count"] = 0

    def _has_cycle_stats(self):
        return any(stats["count"] > 0 for stats in self.schedule_stats.values())

    def apply_mutation(self, schedule):
        mutated = canonicalize_schedule(schedule.clone().long())
        if self.code_len <= 1:
            return mutated

        if random.random() > self.mutation_prob:
            return mutated

        operator_name = self._sample_mutation_operator()
        operator = getattr(self, operator_name)
        mutated = operator(mutated)
        mutated = canonicalize_schedule(mutated)
        return self._limit_group_count(mutated)

    def merge_to_nearby_step(self, schedule):
        if schedule.numel() <= 1:
            return schedule
        idx = random.randrange(schedule.numel())
        old_step = int(schedule[idx].item())
        min_step = max(0, old_step - self.shift_radius)
        max_step = min(int(schedule.max().item()), old_step + self.shift_radius)
        candidates = [step for step in range(min_step, max_step + 1) if step != old_step]
        if not candidates:
            return schedule
        schedule[idx] = random.choice(candidates)
        return schedule

    def local_shift_step(self, schedule):
        idx = random.randrange(schedule.numel())
        old_step = int(schedule[idx].item())
        delta = random.randint(-self.shift_radius, self.shift_radius)
        if delta == 0:
            delta = 1
        new_step = max(0, min(int(schedule.max().item()) + 1, old_step + delta))
        schedule[idx] = new_step
        return schedule

    def block_move(self, schedule):
        if schedule.numel() <= 1:
            return schedule
        block_size = random.randint(1, min(self.block_max_size, schedule.numel()))
        start = random.randint(0, schedule.numel() - block_size)
        delta = random.randint(-self.shift_radius, self.shift_radius)
        if delta == 0:
            delta = 1
        block = schedule[start : start + block_size] + delta
        max_allowed = int(schedule.max().item()) + block_size
        block = torch.clamp(block, min=0, max=max_allowed)
        schedule[start : start + block_size] = block
        return schedule

    def local_swap_order(self, schedule):
        if schedule.numel() <= 2:
            return schedule
        block_size = random.randint(1, max(1, min(self.block_max_size, schedule.numel() // 2)))
        left_start = random.randint(0, schedule.numel() - (2 * block_size))
        right_start = left_start + block_size
        left = schedule[left_start : left_start + block_size].clone()
        right = schedule[right_start : right_start + block_size].clone()
        schedule[left_start : left_start + block_size] = right
        schedule[right_start : right_start + block_size] = left
        return schedule

    def split_group(self, schedule):
        unique_steps = torch.unique(schedule, sorted=True)
        candidate_steps = []
        for step in unique_steps.tolist():
            indices = torch.nonzero(schedule == step, as_tuple=False).flatten()
            if indices.numel() >= 2:
                candidate_steps.append((step, indices))
        if not candidate_steps:
            return schedule
        _, indices = random.choice(candidate_steps)
        split_count = random.randint(1, indices.numel() - 1)
        chosen = indices[torch.randperm(indices.numel())[:split_count]]
        schedule[chosen] = int(schedule.max().item()) + 1
        return schedule

    def crossover(self, parent_a, parent_b):
        parent_a = canonicalize_schedule(parent_a.clone().long())
        parent_b = canonicalize_schedule(parent_b.clone().long())
        if parent_a.numel() <= 1:
            return parent_a
        child = parent_a.clone()
        block_size = random.randint(1, min(self.block_max_size, parent_a.numel()))
        start = random.randint(0, parent_a.numel() - block_size)
        child[start : start + block_size] = parent_b[start : start + block_size]
        child = canonicalize_schedule(child)
        return self._limit_group_count(child)

    def _generate_offspring(self, parents):
        if not parents:
            return []
        offspring = []
        attempts = 0
        max_attempts = max(self.population_size * 6, 8)
        while len(offspring) < self.population_size and attempts < max_attempts:
            attempts += 1
            parent_a = random.choice(parents)["schedule"]
            child = parent_a.clone()
            if len(parents) > 1 and random.random() < self.crossover_prob:
                parent_b = random.choice(parents)["schedule"]
                child = self.crossover(parent_a, parent_b)
            child = self.apply_mutation(child)
            offspring.append(self._candidate_from_schedule(child))
        return offspring

    def _candidate_from_schedule(self, schedule):
        schedule = canonicalize_schedule(schedule.clone().long())
        key = self._schedule_key(schedule)
        stats = self.schedule_stats.get(key)
        if stats is None or not stats["loss_history"]:
            latency = self._fallback_latency(schedule)
            return {
                "key": key,
                "schedule": schedule,
                "loss": math.inf,
                "latency": latency,
                "count": 0,
                "final_loss": math.inf,
                "trend": 0.0,
            }

        loss_history = stats["loss_history"]
        latency_history = stats["latency_history"]
        final_loss = float(loss_history[-1])
        trend = float(loss_history[-1] - loss_history[0]) if len(loss_history) >= 2 else 0.0
        score = self.final_loss_weight * final_loss + self.trend_weight * trend
        latency = float(sum(latency_history) / len(latency_history))
        return {
            "key": key,
            "schedule": stats["schedule"],
            "loss": score,
            "latency": latency,
            "count": int(stats["count"]),
            "final_loss": final_loss,
            "trend": trend,
        }

    def _fallback_latency(self, schedule):
        counts = torch.bincount(schedule, minlength=int(schedule.max().item()) + 1).float()
        num_groups = float(len(counts))
        max_group = float(counts.max().item())
        variance_group = float(((counts - counts.mean()) ** 2).mean().item())
        return num_groups + 0.25 * max_group + 0.05 * variance_group

    def _build_evaluated_population(self):
        candidates = []
        seen = set()
        for schedule in self.population:
            candidate = self._candidate_from_schedule(schedule.detach().cpu())
            if candidate["key"] in seen:
                continue
            seen.add(candidate["key"])
            if math.isfinite(candidate["loss"]):
                candidates.append(candidate)
        return candidates

    def _build_archive(self, candidates):
        if not candidates:
            return []
        finite_candidates = [
            item for item in candidates
            if math.isfinite(item["loss"]) and math.isfinite(item["latency"])
        ]
        if not finite_candidates:
            return []
        fronts = self._nondominated_sort(finite_candidates)
        if not fronts:
            return []
        frontier = fronts[0]
        archive = []
        for idx in frontier:
            item = finite_candidates[idx]
            archive.append(
                {
                    "schedule": item["schedule"].detach().cpu(),
                    "loss": item["loss"],
                    "latency": item["latency"],
                    "count": item.get("count", 0),
                    "final_loss": item.get("final_loss", item["loss"]),
                    "trend": item.get("trend", 0.0),
                }
            )
        archive.sort(key=lambda item: (item["latency"], item["loss"]))
        return archive

    def _limit_candidates(self, candidates, limit):
        if limit <= 0 or len(candidates) <= limit:
            return candidates
        ranked = sorted(
            candidates,
            key=lambda item: (
                not math.isfinite(item["loss"]),
                -int(item.get("count", 0)),
                item.get("final_loss", item["loss"]),
                item["latency"],
            ),
        )
        return ranked[:limit]

    def _deduplicate_candidates(self, candidates):
        dedup = {}
        for item in candidates:
            current = dedup.get(item["key"])
            if current is None:
                dedup[item["key"]] = item
                continue
            current_tuple = (current["loss"], current["latency"], -current.get("count", 0))
            new_tuple = (item["loss"], item["latency"], -item.get("count", 0))
            if new_tuple < current_tuple:
                dedup[item["key"]] = item
        return list(dedup.values())

    def _nsga2_select(self, candidates, target_size):
        if not candidates:
            return []
        fronts = self._nondominated_sort(candidates)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= target_size:
                selected.extend(candidates[idx] for idx in front)
                continue
            crowding = self._crowding_distance(candidates, front)
            ordered = sorted(front, key=lambda idx: crowding[idx], reverse=True)
            remaining = target_size - len(selected)
            selected.extend(candidates[idx] for idx in ordered[:remaining])
            break
        return selected

    def _nondominated_sort(self, candidates):
        domination_sets = {idx: set() for idx in range(len(candidates))}
        dominated_count = {idx: 0 for idx in range(len(candidates))}
        fronts = [[]]

        for i, lhs in enumerate(candidates):
            for j, rhs in enumerate(candidates):
                if i == j:
                    continue
                if self._dominates(lhs, rhs):
                    domination_sets[i].add(j)
                elif self._dominates(rhs, lhs):
                    dominated_count[i] += 1
            if dominated_count[i] == 0:
                fronts[0].append(i)

        front_index = 0
        while front_index < len(fronts) and fronts[front_index]:
            next_front = []
            for idx in fronts[front_index]:
                for dominated_idx in domination_sets[idx]:
                    dominated_count[dominated_idx] -= 1
                    if dominated_count[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            if next_front:
                fronts.append(next_front)
            front_index += 1
        return fronts

    def _crowding_distance(self, candidates, front):
        distances = {idx: 0.0 for idx in front}
        if len(front) <= 2:
            for idx in front:
                distances[idx] = math.inf
            return distances

        objectives = ["loss", "latency"]
        for objective in objectives:
            ordered = sorted(front, key=lambda idx: candidates[idx][objective])
            distances[ordered[0]] = math.inf
            distances[ordered[-1]] = math.inf
            min_value = candidates[ordered[0]][objective]
            max_value = candidates[ordered[-1]][objective]
            if max_value == min_value:
                continue
            scale = max_value - min_value
            for position in range(1, len(ordered) - 1):
                prev_value = candidates[ordered[position - 1]][objective]
                next_value = candidates[ordered[position + 1]][objective]
                distances[ordered[position]] += (next_value - prev_value) / scale
        return distances

    @staticmethod
    def _dominates(lhs, rhs):
        return (
            lhs["loss"] <= rhs["loss"]
            and lhs["latency"] <= rhs["latency"]
            and (lhs["loss"] < rhs["loss"] or lhs["latency"] < rhs["latency"])
        )

    def _sample_mutation_operator(self):
        operators = list(self.mutation_weights.keys())
        weights = list(self.mutation_weights.values())
        return random.choices(operators, weights=weights, k=1)[0]

    def _limit_group_count(self, schedule):
        schedule = canonicalize_schedule(schedule)
        unique_steps = torch.unique(schedule, sorted=True)
        if unique_steps.numel() <= self.max_groups:
            return schedule
        keep = unique_steps[: self.max_groups - 1].tolist()
        overflow = unique_steps[self.max_groups - 1 :].tolist()
        collapsed = schedule.clone()
        target_step = keep[-1] if keep else 0
        for step in overflow:
            collapsed[collapsed == step] = target_step
        return canonicalize_schedule(collapsed)

    @staticmethod
    def _schedule_key(schedule):
        return tuple(schedule.tolist())

    def state_dict(self):
        return {
            "population": [item.detach().cpu() for item in self.population],
            "archive": [
                {
                    "schedule": item["schedule"].detach().cpu(),
                    "loss": item["loss"],
                    "latency": item["latency"],
                    "count": item.get("count", 0),
                    "final_loss": item.get("final_loss", item["loss"]),
                    "trend": item.get("trend", 0.0),
                }
                for item in self.archive
            ],
            "schedule_stats": {
                key: {
                    "schedule": value["schedule"].detach().cpu(),
                    "loss_history": list(value["loss_history"]),
                    "latency_history": list(value["latency_history"]),
                    "count": value["count"],
                }
                for key, value in self.schedule_stats.items()
            },
        }

    def load_state_dict(self, state_dict):
        population = state_dict.get("population")
        archive = state_dict.get("archive")
        schedule_stats = state_dict.get("schedule_stats", {})
        if population:
            self.population = [item.to(self.device) for item in population]
        if archive:
            self.archive = archive
        if schedule_stats:
            restored = {}
            for key, value in schedule_stats.items():
                restored[key] = {
                    "schedule": value["schedule"],
                    "loss_history": list(value.get("loss_history", [])),
                    "latency_history": list(value.get("latency_history", [])),
                    "count": value.get("count", 0),
                }
            self.schedule_stats = restored

    def archive_summary(self):
        if not self.archive:
            return {"size": 0, "best_loss": None, "best_latency": None, "avg_groups": None}
        return {
            "size": len(self.archive),
            "best_loss": min(item["loss"] for item in self.archive),
            "best_latency": min(item["latency"] for item in self.archive),
            "avg_groups": sum(int(item["schedule"].max().item()) + 1 for item in self.archive) / len(self.archive),
        }


def gather_records_to_rank0(records, dst=0):
    if not dist.is_available() or not dist.is_initialized():
        return records if dst == 0 else None
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [None for _ in range(world_size)] if rank == dst else None
    dist.gather_object(records, object_gather_list=gathered, dst=dst)
    if rank != dst:
        return None
    merged = {}
    for shard in gathered:
        if not shard:
            continue
        for record in shard:
            schedule = canonicalize_schedule(record["schedule"].detach().cpu().long())
            key = ScheduleManager._schedule_key(schedule)
            merged_record = merged.get(key)
            if merged_record is None:
                merged_record = {
                    "schedule": schedule,
                    "loss_sum": 0.0,
                    "latency_sum": 0.0,
                    "count": 0,
                }
                merged[key] = merged_record
            merged_record["loss_sum"] += float(record.get("loss_sum", record.get("loss", 0.0)))
            merged_record["latency_sum"] += float(record.get("latency_sum", record.get("latency", 0.0)))
            merged_record["count"] += int(record.get("count", 1))
    return list(merged.values())


def broadcast_schedule_manager_state(schedule_manager, src=0):
    if not dist.is_available() or not dist.is_initialized():
        return
    obj = [schedule_manager.state_dict() if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj, src=src)
    if dist.get_rank() != src:
        schedule_manager.load_state_dict(obj[0])
