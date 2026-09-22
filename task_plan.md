# Action-Structured Self-Conditioned Diffusion Policy

## Objective

Implement the specified workpoint-one method in phases while preserving the existing scheduler, training target, horizons, optimizer, and observation encoder. The five ablations must share one code path.

## Phase status

| Phase | Scope | Status | Acceptance condition |
|---|---|---|---|
| 0 | Baseline audit; no behavior change | complete | Recorded code paths, tensor shapes, checkpoint/EMA/config facts, and minimal baseline loss/inference evidence. |
| 1 | Self-conditioning only | complete | `original_dp` and `sc_only` share the core path; two-pass training and diffusion-step feedback work. |
| 2 | Position-wise feedback gate | pending_user_confirmation | Gate emits `[B,H,1]`, logs statistics, and supports SC-equivalent debug behavior. |
| 3 | Feedback perturbation | pending_user_confirmation | Training-only feedback perturbation is independently configurable and is identity during evaluation. |
| 4 | Unified ablation interface | pending_user_confirmation | Five configurations use common training/evaluation entry points and reproducible snapshots. |

## Constraints

- Do not change scheduler mathematics, noise-prediction target, optimizer, horizons, or observation encoder.
- Do not use Push-T as the core task, ground-truth actions as feedback, or executed actions from the previous robot-control step.
- Stop after each phase and await the user's confirmation before the next implementation phase.

## Current next action

Await user confirmation before Phase 2. Closed-loop `original_dp` versus `sc_only` benchmark training awaits the RoboMimic Can dataset.
