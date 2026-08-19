import torch

from refa_edge.data import OPPOSE, QUERY, SUPPORT, RelationalRevisionDataset, TaskSpec


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
