import torch

from refa_edge.data import (
    OPPOSE,
    QUERY,
    SUPPORT,
    KeyedParityDataset,
    KeyedParitySpec,
    RelationalRevisionDataset,
    TaskSpec,
)


def test_dataset_is_balanced_and_query_is_last() -> None:
    dataset = RelationalRevisionDataset(
        30,
        TaskSpec(seq_len=12, num_entities=20, num_relations=5),
        seed=3,
    )
    counts = torch.bincount(dataset.labels, minlength=3)
    assert counts.tolist() == [10, 10, 10]
    assert torch.all(dataset.events[:, -1, 3] == QUERY)
    assert torch.all(dataset.events[:, -1, 4] == 1)


def test_latest_matching_evidence_agrees_with_known_label() -> None:
    dataset = RelationalRevisionDataset(
        60,
        TaskSpec(
            seq_len=14,
            num_entities=20,
            num_relations=5,
            revision_probability=1.0,
        ),
        seed=9,
    )
    for events, label in dataset:
        query = events[-1, :3]
        matches = torch.all(events[:-1, :3] == query, dim=-1)
        if int(label) in (SUPPORT, OPPOSE):
            assert matches.any()
            assert int(events[:-1][matches][-1, 3]) == int(label)
        else:
            assert not matches.any()


def test_keyed_parity_has_no_polarity_leakage_and_matches_counts() -> None:
    dataset = KeyedParityDataset(
        40,
        KeyedParitySpec(
            history_len=16,
            num_entities=24,
            num_relations=5,
            num_active_keys=4,
        ),
        seed=11,
    )
    assert torch.bincount(dataset.labels, minlength=2).tolist() == [20, 20]
    assert torch.all(dataset.events[:, :-1, 3] == SUPPORT)
    assert torch.all(dataset.events[:, :-1, 4] == 0)
    assert torch.all(dataset.events[:, -1, 3] == QUERY)
    assert torch.all(dataset.events[:, -1, 4] == 1)
    for events, label in dataset:
        query = events[-1, :3]
        target_count = torch.all(events[:-1, :3] == query, dim=-1).sum()
        assert int(target_count % 2) == int(label)
        assert torch.unique(events[:-1, :3], dim=0).shape[0] == 4
