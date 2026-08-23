# A1/K3 preregistration — keyed parity and the spectral write gate

Status: protocol frozen before confirmatory seeds 101–110 are inspected.

## Why the original queued task is not valid

The handoff proposed alternating SUPPORT/OPPOSE events with the final label equal to
parity. That construction leaks the answer through the last polarity: an alternating
chain's last event already identifies whether its length is odd or even. In full REFA,
the GRU workspace supplies a second confound because it can count independently of the
fast memory. A positive result on that task would therefore not identify the proposed
negative-eigenvalue mechanism.

## Corrected claim and task

Claim: allowing the key-direction transition eigenvalue into `[-1, 0)` gives REFA's
fast-memory path a period-2 state that the current and proximal-safe gates lack, and
that state supports length extrapolation on associative parity.

Each example contains 16 history writes during training and 64 at OOD test. Two keys
from a 16-subject vocabulary are interleaved, with relation and object held constant so
the probe does not simultaneously test additive triple binding. Every history event uses the identical SUPPORT
polarity; the queried key appears only in the final query row. The binary target is the
parity of the number of writes to that key. Thus neither event polarity nor the final
history token contains the label.

The primary endpoint uses a memory-only diagnostic readout. It has access to the final
support/opposition recall vectors, the query's expected value vector, their norms, and
the two recall-on-query-value projection coefficients, but not the GRU workspace. The
query-aligned coefficient is required to identify amplitude across identity-dependent
value directions; omitting it made development seed 1701 stay at chance even after the
expressive gate entered the negative regime. A second development check reached only
0.627 ID because the random triple space also tested compositional binding, so the key
vocabulary was reduced to subject identity. Both corrections were made before any
confirmatory seed was run. The full REFA readout is secondary and measures whether the
mechanism survives composition with the architecture.

## Fixed interventions

- `current`: `beta = sigmoid(g)`.
- `safe`: `a = softplus(g); beta = a / (1 + a)`.
- `expressive`: `beta_h = (1 + lambda_h) sigmoid(g)`.
- Effective per-bank rate is `beta_h r_h`; all modes use the same initialization,
  optimizer, data, parameter count, and seed set.
- One bank is used in this mechanism-isolation experiment so soft routing cannot make
  the expressive interval unreachable. Multi-bank translation is a required follow-up,
  not part of this claim.

For normalized key `k`, the transition spectrum is
`{lambda_h - beta_h r_h, lambda_h}`. Every evaluation logs the minimum eigenvalue,
negative-eigenvalue fraction, spectral radius, effective gate, decay, and routing
statistics. A runtime certificate requires spectral radius `<= 1 + 1e-6`.

## Confirmatory protocol

- Configuration: `configs/parity_kill.yaml`.
- Seeds: 101–110.
- Conditions: three gate modes × memory-only and full readouts = 60 runs.
- Training: 2,048 examples, 30 epochs, AdamW, learning rate 1e-3, batch 64, no
  weight decay, gradient clipping at 1.0.
- ID validation: 512 independently generated length-16 examples.
- OOD test: 1,024 independently generated length-64 examples.
- Report medians and IQRs; retain per-seed raw JSON and accuracy by queried-key count.
- Development seed 1701 may be used only to catch implementation or optimization
  failures before the confirmatory run. Any protocol change after that smoke run must
  be recorded here before seeds 101–110 are evaluated.

Development amendments, all before confirmatory seeds: the initial random-triple probe
was replaced by the finite subject vocabulary described above. With four active keys,
rank 8, and 30 diagnostic epochs, expressive gating reached only about 0.61 ID accuracy
despite a strongly negative spectrum, implicating key interference. One final isolation
check therefore used two active keys, rank 16, and 30 epochs. It reached 0.971 ID and
0.891 length-64 accuracy for expressive gating while both controls remained at chance.
That exact regime is now frozen for confirmation; no further task redesign is permitted.

## Decision rule

The primary result supports the practical spectral-parity hypothesis only if:

1. the expressive memory-only condition engages the mechanism (median minimum
   eigenvalue `<= -0.5` and median negative-eigenvalue fraction `>= 0.10`);
2. its median length-64 accuracy is at least 0.80; and
3. it exceeds the better of current/safe by at least 0.15 median accuracy.

If the negative spectrum is engaged without meeting the accuracy and superiority
thresholds, the result does not support the hypothesis. If the expressive gate does
not enter the preregistered spectrum, the run is inconclusive about expressivity and is
reported as an optimization/mechanism-engagement failure, not rescued as a positive.

## Scope

A positive result would establish a small mechanistic capability cliff, not show that
typed opposition banks, memory-first AI, or language-scale transfer works. A negative
result would kill this proposed parity experiment/gate pairing, not the independent
measured-uncertainty or provenance theses.
