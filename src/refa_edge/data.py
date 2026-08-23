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


@dataclass(frozen=True)
class KeyedParitySpec:
    """Specification for the associative parity mechanism probe.

    ``history_len`` counts writes and excludes the final query row. Every write uses
    the same polarity token, so neither the last token nor a polarity embedding leaks
    the parity label.
    """

    history_len: int
    num_entities: int
    num_relations: int
    num_active_keys: int = 4


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


class KeyedParityDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Balanced keyed event-count parity with a query revealed at the end.

    A sample contains ``num_active_keys`` distinct relational identities. The history
    is a shuffled stream of identical SUPPORT writes to those identities. The final
    row asks about one identity, and the label is the parity of that identity's write
    count (SUPPORT/even, OPPOSE/odd). Because all history symbols share one polarity and
    the queried identity is revealed only after the history, a model must retain a
    keyed period-2 state rather than read the answer from the final history event.
    """

    def __init__(self, size: int, spec: KeyedParitySpec, seed: int = 0) -> None:
        self.size = int(size)
        self.spec = spec
        self.seed = int(seed)
        if spec.history_len < spec.num_active_keys + 1:
            raise ValueError("history_len must leave room for every active key and both parities")
        key_space = spec.num_entities
        if spec.num_active_keys > key_space:
            raise ValueError("num_active_keys exceeds the number of distinct relational keys")
        examples = [self._generate(index) for index in range(self.size)]
        self.events = torch.stack([item[0] for item in examples])
        self.labels = torch.tensor([item[1] for item in examples], dtype=torch.long)
        self.target_counts = torch.tensor([item[2] for item in examples], dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.events[index], self.labels[index]

    def _randint(self, high: int, generator: torch.Generator) -> int:
        return int(torch.randint(high, (1,), generator=generator).item())

    def _key(self, generator: torch.Generator) -> tuple[int, int, int]:
        # Hold relation and object constant: this experiment isolates keyed period-2
        # state and must not simultaneously test additive triple binding.
        return (self._randint(self.spec.num_entities, generator), 0, 0)

    def _generate(self, index: int) -> tuple[torch.Tensor, int, int]:
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * index)
        label = index % 2
        keys: list[tuple[int, int, int]] = []
        while len(keys) < self.spec.num_active_keys:
            candidate = self._key(generator)
            if candidate not in keys:
                keys.append(candidate)

        target_index = self._randint(self.spec.num_active_keys, generator)
        max_target_count = self.spec.history_len - (self.spec.num_active_keys - 1)
        valid_counts = [
            count
            for count in range(1, max_target_count + 1)
            if count % 2 == label
        ]
        target_count = valid_counts[self._randint(len(valid_counts), generator)]

        key_indices = [target_index] * target_count
        other_indices = [
            key_index
            for key_index in range(self.spec.num_active_keys)
            if key_index != target_index
        ]
        # Give every distractor key at least one write, then allocate the remainder.
        key_indices.extend(other_indices)
        remainder = self.spec.history_len - len(key_indices)
        for _ in range(remainder):
            key_indices.append(other_indices[self._randint(len(other_indices), generator)])
        permutation = torch.randperm(self.spec.history_len, generator=generator).tolist()
        rows = [[*keys[key_indices[position]], SUPPORT, 0] for position in permutation]
        rows.append([*keys[target_index], QUERY, 1])
        return torch.tensor(rows, dtype=torch.long), label, target_count


def task_spec_from_config(config: dict) -> TaskSpec:
    return TaskSpec(
        seq_len=int(config["seq_len"]),
        num_entities=int(config["num_entities"]),
        num_relations=int(config["num_relations"]),
        revision_probability=float(config.get("revision_probability", 0.7)),
    )


def keyed_parity_spec_from_config(config: dict, history_len: int) -> KeyedParitySpec:
    return KeyedParitySpec(
        history_len=int(history_len),
        num_entities=int(config["num_entities"]),
        num_relations=int(config["num_relations"]),
        num_active_keys=int(config.get("num_active_keys", 4)),
    )
