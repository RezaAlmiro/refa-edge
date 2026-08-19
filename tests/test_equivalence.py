from refa_edge.benchmarks.equivalence import check_dense_stream_equivalence


def test_dense_and_streaming_fast_weights_are_equivalent() -> None:
    report = check_dense_stream_equivalence()
    assert report["passed"], report
