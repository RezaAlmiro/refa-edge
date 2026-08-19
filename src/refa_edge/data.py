from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

SUPPORT = 0
OPPOSE = 1
QUERY = 2

LABEL_NAMES = {
    SUPPORT: "supported",
    OPPOSE: "opposed",
    QUERY: "unknown",
}


@dataclass(frozen=True)
class TaskSpec:
    seq_len: int
    num_entities: int
    num_relations: int
    revision_probability: float = 0.7


class RelationalRevisionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Balanced evidence streams where the latest matching event determines the answer.

    Each row is [subject, relation, object, polarity, is_query]. The final row is
    always the query. Label 0 means supported, 1 means opposed, and 2 means unknown.
    Samples are generated independently, so no memory crosses a sample boundary.
    """

    def __init__(self, size: int, spec: TaskSpec, seed: int = 0) -> None:
        self.size = int(size)
        self.spec = spec
        self.seed = int(seed)
        examples = [self._generate(index) for index in range(self.size)]
        self.events = torch.stack([item[0] for item in examples])
        self.labels = torch.tensor([item[1] for item in examples], dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.events[index], self.labels[index]

    def _randint(self, high: int, generator: torch.Generator) -> int:
        return int(torch.randint(high, (1,), generator=generator).item())

    def _distractor(
        self,
        target: tuple[int, int, int],
        generator: torch.Generator,
    ) -> list[int]:
        while True:
            subject = self._randint(self.spec.num_entities, generator)
            relation = self._randint(self.spec.num_relations, generator)
            obj = self._randint(self.spec.num_entities, generator)
            if (subject, relation, obj) != target:
                polarity = self._randint(2, generator)
                return [subject, relation, obj, polarity, 0]

    def _generate(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * index)
        label = index % 3
        query = (
            self._randint(self.spec.num_entities, generator),
            self._randint(self.spec.num_relations, generator),
            self._randint(self.spec.num_entities, generator),
        )
        history_length = self.spec.seq_len - 1
        rows = [self._distractor(query, generator) for _ in range(history_length)]

        if label in (SUPPORT, OPPOSE):
            final_slot = history_length - 1 - self._randint(min(4, history_length), generator)
            rows[final_slot] = [query[0], query[1], query[2], label, 0]

            can_revise = final_slot > 0 and torch.rand((), generator=generator).item() < (
                self.spec.revision_probability
            )
            if can_revise:
                earlier_slot = self._randint(final_slot, generator)
                rows[earlier_slot] = [query[0], query[1], query[2], 1 - label, 0]
        else:
            # A close decoy forces the model to bind all three identity fields.
            decoy_slot = self._randint(history_length, generator)
            decoy_object = (query[2] + 1 + self._randint(self.spec.num_entities - 1, generator))
            decoy_object %= self.spec.num_entities
            rows[decoy_slot] = [query[0], query[1], decoy_object, self._randint(2, generator), 0]

        rows.append([query[0], query[1], query[2], QUERY, 1])
        return torch.tensor(rows, dtype=torch.long), label


def task_spec_from_config(config: dict) -> TaskSpec:
    return TaskSpec(
        seq_len=int(config["seq_len"]),
        num_entities=int(config["num_entities"]),
        num_relations=int(config["num_relations"]),
        revision_probability=float(config.get("revision_probability", 0.7)),
    )
