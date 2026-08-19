# REFA Edge

**Relational Evidence Field Architecture — a local-first, low-resource research MVP.**

[![Tests](https://github.com/RezaAlmiro/refa-edge/actions/workflows/tests.yml/badge.svg)](https://github.com/RezaAlmiro/refa-edge/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

REFA Edge is an experimental alternative to treating every problem as next-token
prediction over one large, static parameter store. It represents experience as
relational evidence events, keeps support and opposition separate, revises memory
when later evidence addresses the same content, and forms an answer in a small
recurrent workspace.

The repository is designed to answer a concrete question:

> Can a useful relational-memory core be trained and run locally on an older laptop,
> while being compared fairly with conventional small models?

## Read this first

This is a **reference implementation and falsifiable experiment**, not a peer-reviewed
result and not a frontier language model. It does not claim exceptional accuracy.
The included smoke run proves that the pipeline works; only measurements produced on
your machine belong in a results table.

An RTX 2060 with 6 GB VRAM and 32 GB system RAM is a reasonable target for the
included small configurations. It is not enough to train a competitive general
language model from scratch. REFA Edge initially targets relational recall, revision,
and contrastive evidence tasks where its memory structure can be tested directly.

## What is included

| Component | Purpose |
|---|---|
| <code>refa</code> | Typed support/opposition fast-memory banks plus recurrent workspace |
| <code>gru</code> | Conventional recurrent baseline |
| <code>transformer</code> | Small causal Transformer baseline |
| <code>fast_dense</code> | Quadratic dense fast-weight reference |
| <code>fast_stream</code> | Exact linear-state rewrite of the dense reference |
| External adapter | Load another PyTorch model through <code>module:function</code> |
| Benchmark harness | Same data, split, optimizer protocol, metrics, seeds, and output schema |
| Local REST API | Serve any saved built-in checkpoint on your own machine |

~~~mermaid
flowchart TD
    A["Evidence events"] --> B["Relation router"]
    B --> C["Support banks"]
    B --> D["Opposition banks"]
    C --> E["Contrastive recall"]
    D --> E
    E --> F["Recurrent hypothesis workspace"]
    F --> G["Supported / opposed / unknown"]
~~~

## Five-minute start

You need 64-bit Python 3.10 or 3.11, Git, and roughly 5 GB of free disk space if
PyTorch must be downloaded.

### Windows PowerShell

~~~powershell
git clone https://github.com/RezaAlmiro/refa-edge.git
cd refa-edge
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/refa-edge.exe doctor
.venv/Scripts/refa-edge.exe smoke
~~~

Using the virtual environment's executable directly avoids PowerShell activation
policy problems.

### Linux or macOS

~~~bash
git clone https://github.com/RezaAlmiro/refa-edge.git
cd refa-edge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
refa-edge doctor
refa-edge smoke
~~~

The smoke command trains four tiny models and writes:

- <code>runs/smoke/results.json</code> — complete machine-readable record
- <code>runs/smoke/results.csv</code> — easy spreadsheet comparison
- <code>runs/smoke/refa.pt</code> — trained REFA checkpoint

## Use the RTX 2060 profile

First run <code>refa-edge doctor</code>. It must say <code>CUDA visible: True</code>.
If it says false, install the current CUDA-enabled PyTorch build using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then run
the doctor again.

~~~powershell
.venv/Scripts/refa-edge.exe fit-check --config configs/rtx2060_6gb.yaml --model refa
.venv/Scripts/refa-edge.exe benchmark --config configs/rtx2060_6gb.yaml --models refa,gru,transformer,fast_stream
~~~

Laptop RTX 2060 models differ. If memory runs out, change
<code>train.batch_size</code> from 16 to 8 in the YAML file. Do not quietly change one
model's batch size during a comparison; copy the config and record the change.

## Compare a different model

An external model only needs to accept an integer tensor shaped
<code>[batch, sequence, 5]</code> and return three class logits. The example adapter
is runnable:

~~~bash
refa-edge benchmark \
  --config configs/smoke.yaml \
  --models refa,gru \
  --external-factory refa_edge.examples.external_model:build_model \
  --external-name mean_pool
~~~

See [Benchmarking](docs/BENCHMARKING.md) for the interface, controls, and rules for a
credible comparison.

## Run the local REST API

Install the optional API packages after training:

~~~bash
python -m pip install -e ".[api]"
refa-edge serve --checkpoint runs/smoke/refa.pt
~~~

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The interactive page
lets you send evidence and a query without writing client code. The server binds to
your own machine only by default.

## Verification commands

~~~bash
python -m pip install -e ".[dev]"
pytest
refa-edge check-equivalence
~~~

The equivalence gate checks both outputs and gradients for the dense and streaming
fast-weight calculations in float64. A change that breaks the algebra fails the test.

## Documentation

- [Beginner setup and troubleshooting](docs/GETTING_STARTED.md)
- [Architecture, equations, boundaries, and roadmap](docs/ARCHITECTURE.md)
- [Relationship to Pathway BDH and Soup](docs/RELATED_WORK.md)
- [Fair benchmarking and external adapters](docs/BENCHMARKING.md)
- [Blank results card](docs/RESULTS_TEMPLATE.md)

## Provenance and scope

This is an independent clean implementation. It is informed by the public
[Pathway BDH repository](https://github.com/pathwaycom/bdh) and by systems techniques
demonstrated in [Soup](https://github.com/MakazhanAlpamys/Soup). No source code was
copied from either project. See [NOTICE](NOTICE.md) for the precise boundary.

## License

MIT. Research findings produced with this code should include the configuration,
seed, raw JSON result, hardware, and commit hash.
