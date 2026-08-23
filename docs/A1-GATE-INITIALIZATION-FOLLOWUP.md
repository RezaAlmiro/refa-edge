# A1 post-hoc follow-up — access to the negative-spectrum basin

Status: protocol frozen before seeds 201–210 are run. This is a post-hoc mechanistic
follow-up, not part of the confirmatory result in `A1-KEYED-PARITY-PREREGISTRATION.md`.

## Observation that motivates the follow-up

The preregistered expressive memory-only condition was bimodal. Five of ten seeds drove
the transition to approximately -1 and reached 0.879–0.951 length-64 accuracy. Five
stayed in the positive-spectrum regime and remained at chance. Current and safe controls
remained at chance. The frozen aggregate verdict is therefore inconclusive because the
mechanism was not reliably engaged.

## Hypothesis

The expressive parameterization contains the required period-2 solution, but default
linear-layer initialization places the raw gate near zero: with `lambda = 0.995`,
`sigmoid(0)(1 + lambda)` gives a transition near zero, not a reflection. Gradient descent
then enters either a positive fixed-point basin or a negative reflection basin.

Set the raw write-gate bias to 2.0 in every gate condition. This value is derived before
running the experiment from the spectrum, not tuned on a result:

- expressive initial eigenvalue:
  `lambda - (1 + lambda) sigmoid(2) ~= -0.762`;
- current initial eigenvalue: `lambda - sigmoid(2) ~= 0.114`;
- safe initial eigenvalue:
  `lambda - softplus(2)/(1 + softplus(2)) ~= 0.315`.

Thus the identical raw initialization makes only the expressive parameterization start
inside the negative-spectrum basin.

## Fixed protocol and decision rule

Everything except the shared raw gate bias and fresh seeds is identical to the frozen A1
configuration. Configuration: `configs/parity_gate_init_followup.yaml`. Seeds: 201–210.
Conditions: three gates × memory-only/full readouts. Report medians and IQRs.

The optimization-access hypothesis is supported on the primary memory-only endpoint if:

1. all ten expressive seeds finish with a negative-eigenvalue fraction at least 0.90;
2. expressive median length-64 accuracy is at least 0.80 with Q25 at least 0.75; and
3. expressive exceeds the better control median by at least 0.15.

The full readout is a separate translation endpoint. It supports translation into the
current REFA composition only if expressive median length-64 accuracy is at least 0.80;
failure there does not negate the isolated mechanism but blocks an architecture claim.

No hyperparameter or task change is permitted after the first seed begins. A positive
result establishes a spectral mechanism plus an initialization requirement—not typed
opposition memory, language transfer, or general memory-first intelligence.
