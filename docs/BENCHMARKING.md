# Benchmarking without fooling ourselves

The harness makes comparisons easy; scientific validity still depends on how the
experiment is designed.

## Included task

Each sample contains a sequence of five-field events:

<code>[subject, relation, object, polarity, is_query]</code>

The final event is a query. The balanced labels are:

0. supported;
1. opposed;
2. unknown.

For known answers, the latest exact matching evidence determines the label. An
earlier matching event often has the opposite stance, testing revision. Unknown
samples contain close distractors, testing whether all identity fields are bound.

Train and validation samples are generated from different deterministic seed ranges.
Samples never share a recurrent state, so there is no cross-example memory leakage.

## One-command comparison

~~~bash
refa-edge benchmark \
  --config configs/rtx2060_6gb.yaml \
  --models refa,gru,transformer,fast_dense,fast_stream
~~~

All selected models receive:

- the same generated train and validation samples;
- the same batch order seed;
- the same epoch count, optimizer type, learning rate, clipping, and batch size;
- a fresh deterministic initialization;
- the same metric implementation.

The run reports parameter count rather than pretending different architectures have
identical capacity. For a parameter-matched study, adjust widths in separate config
files until counts are close, and disclose every config.

## Metrics

| Metric | Use | Limitation |
|---|---|---|
| Validation accuracy | Primary correctness check | Hides calibration and class-specific errors |
| Validation loss | Confidence-sensitive check | Hard to explain alone |
| Parameters | Learned storage size | Does not measure activation memory |
| Examples/second | Training throughput | Hardware- and software-dependent |
| Peak CUDA MiB | Allocated tensor memory | Excludes some driver and non-PyTorch memory |
| Train seconds | Wall-clock cost | Includes implementation efficiency |

The JSON file is the record of truth. A README table should be generated from raw
files, never typed from memory.

## Dense versus streaming baseline

The two fast-weight baselines implement:

$$
y_t = \sum_{s<t}(q_t^\top k_s)v_s
$$

The streamed form stores:

$$
S_t = \sum_{s<t} k_s v_s^\top,
\qquad
y_t = q_t^\top S_t
$$

Run:

~~~bash
refa-edge check-equivalence
~~~

It compares both outputs and gradients in float64. The streamed implementation should
not be called “equivalent” if this gate fails.

## Plug in another PyTorch model

Create a factory:

~~~python
from torch import nn


def build_model(task_config: dict, model_config: dict) -> nn.Module:
    return MyModel(
        num_entities=task_config["num_entities"],
        num_relations=task_config["num_relations"],
        d_model=model_config["d_model"],
    )
~~~

The model contract is:

- input: integer tensor <code>[batch, sequence, 5]</code>;
- output: floating tensor <code>[batch, 3]</code>;
- no internal optimizer step;
- no access to validation labels or validation generation seed.

Load it without modifying the harness:

~~~bash
refa-edge benchmark \
  --config configs/smoke.yaml \
  --models refa,gru \
  --external-factory my_package.my_module:build_model \
  --external-name my_model
~~~

The bundled <code>refa_edge.examples.external_model</code> is a complete working
adapter.

## Comparing an incompatible language model

A text-generating model cannot be dropped into this event-classification interface
without an adapter. The adapter must define a stable textual encoding for events and
map its answer back to the three labels. Disclose prompt text, decoding settings,
tokenizer, precision, context length, and whether any examples appear in the prompt.

Do not compare a pretrained language model with a from-scratch REFA model as though
their training compute and prior data were equal. Report two distinct questions:

1. practical performance using each model as available;
2. architectural learning efficiency under controlled training data and compute.

## Minimum credible result

Before publishing a capability claim:

1. Freeze a commit and save its hash.
2. Choose the metric and hypothesis before inspecting final test results.
3. Use at least five seeds.
4. Report mean, standard deviation, and every raw JSON file.
5. Match parameter count or report the difference prominently.
6. Include failure cases and unknown-class performance.
7. Test longer sequences than training sequences.
8. Separate measured facts from explanations and speculation.

Use [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md) as the public result card.
