import random

import torch
import torch.distributed as dist

from autoregressive.train.mask_builder import canonicalize_schedule, make_fixed_schedule


class ScheduleManager:
    def __init__(
        self,
        code_len,
        num_groups,
        evolve_every,
        population_size,
        mutation_prob,
        max_groups=None,
        device=None,
    ):
        self.code_len = code_len
        self.num_groups = num_groups
        self.evolve_every = max(int(evolve_every), 0)
        self.population_size = max(int(population_size), 1)
        self.mutation_prob = float(mutation_prob)
        self.max_groups = max_groups if max_groups is not None else max(num_groups, 1)
        self.device = device
        self.population = []
        self.archive = []
        self.pending_records = []
        self._initialize_population()

    def _initialize_population(self):
        base = make_fixed_schedule(self.code_len, self.num_groups, device=self.device)
        self.population = [base.clone()]
        while len(self.population) < self.population_size:
            self.population.append(self.mutate(base))

    def sample(self, batch_size, step, device=None):
        device = device or self.device
        schedules = []
        for _ in range(batch_size):
            selected = self.population[random.randrange(len(self.population))]
            schedules.append(selected.clone())
        return torch.stack(schedules, dim=0).to(device)

    def record(self, schedule_steps, sample_loss, latency_proxy):
        for idx in range(schedule_steps.shape[0]):
            self.pending_records.append(
                {
                    "schedule": schedule_steps[idx].detach().cpu().long(),
                    "loss": float(sample_loss[idx].detach().cpu().item()),
                    "latency": float(latency_proxy[idx].detach().cpu().item()),
                }
            )

    def should_evolve(self, step):
        return self.evolve_every > 0 and step > 0 and step % self.evolve_every == 0

    def evolve_if_needed(self, step):
        if not self.should_evolve(step) or not self.pending_records:
            return False
        self._update_archive(self.pending_records)
        self.pending_records = []
        seeds = self.archive[: self.population_size]
        if not seeds:
            return False
        new_population = []
        for item in seeds:
            new_population.append(item["schedule"].to(self.device))
        while len(new_population) < self.population_size:
            parent = random.choice(seeds)["schedule"].to(self.device)
            new_population.append(self.mutate(parent))
        self.population = new_population
        return True

    def mutate(self, schedule):
        schedule = canonicalize_schedule(schedule.clone().long())
        if self.code_len <= 1:
            return schedule
        mutated = schedule.clone()
        num_mutations = 0
        for idx in range(self.code_len):
            if random.random() < self.mutation_prob:
                mutated[idx] = random.randint(0, max(self.max_groups - 1, 0))
                num_mutations += 1
        if num_mutations == 0:
            idx = random.randrange(self.code_len)
            mutated[idx] = random.randint(0, max(self.max_groups - 1, 0))
        return canonicalize_schedule(mutated)

    def _update_archive(self, records):
        merged = self.archive + records
        frontier = []
        for candidate in merged:
            dominated = False
            for other in merged:
                if other is candidate:
                    continue
                if self._dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)

        dedup = {}
        for item in frontier:
            key = tuple(item["schedule"].tolist())
            current = dedup.get(key)
            if current is None or (item["loss"], item["latency"]) < (current["loss"], current["latency"]):
                dedup[key] = item
        self.archive = sorted(dedup.values(), key=lambda item: (item["latency"], item["loss"]))

    @staticmethod
    def _dominates(lhs, rhs):
        return (
            lhs["loss"] <= rhs["loss"]
            and lhs["latency"] <= rhs["latency"]
            and (lhs["loss"] < rhs["loss"] or lhs["latency"] < rhs["latency"])
        )

    def state_dict(self):
        return {
            "population": [item.detach().cpu() for item in self.population],
            "archive": [
                {
                    "schedule": item["schedule"].detach().cpu(),
                    "loss": item["loss"],
                    "latency": item["latency"],
                }
                for item in self.archive
            ],
        }

    def load_state_dict(self, state_dict):
        population = state_dict.get("population")
        archive = state_dict.get("archive")
        if population:
            self.population = [item.to(self.device) for item in population]
        if archive:
            self.archive = archive

    def archive_summary(self):
        if not self.archive:
            return {"size": 0, "best_loss": None, "best_latency": None}
        return {
            "size": len(self.archive),
            "best_loss": min(item["loss"] for item in self.archive),
            "best_latency": min(item["latency"] for item in self.archive),
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
    merged = []
    for shard in gathered:
        if shard:
            merged.extend(shard)
    return merged


def broadcast_schedule_manager_state(schedule_manager, src=0):
    if not dist.is_available() or not dist.is_initialized():
        return
    obj = [schedule_manager.state_dict() if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj, src=src)
    if dist.get_rank() != src:
        schedule_manager.load_state_dict(obj[0])
