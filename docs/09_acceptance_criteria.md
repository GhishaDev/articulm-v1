# Acceptance Criteria

M1 — Code complete:
- config tests
- schema tests
- model forward tests
- loss tests
- backward tests
- checkpoint tests

M2 — Tiny overfit:
- no NaN/Inf
- correct masks
- Viseme loss strongly decreases
- Strength loss strongly decreases
- save/load parity

M3 — First synthetic training:
- valid checkpoint
- curves
- Viseme accuracy/macro-F1
- Strength MAE/RMSE
- per-class/slice report
- inference sample

M4 — Product-oriented evaluation:
- phrase boundaries
- numbers
- named entities
- tone/sandhi
- long broadcast sentences

M5 — Human Gold readiness:
- fixed Human Gold validation/test
- perceptual protocol
- separate synthetic vs Human reports

Reject a run if:
- train/val leakage exists
- teacher labels enter encoder input
- raw value is treated as true Strength without justification
- padding contributes to loss
- resume is broken
- inference schema differs from training
