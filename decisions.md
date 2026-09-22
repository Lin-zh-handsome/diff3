# Implementation decisions

| Decision | Status | Rationale |
|---|---|---|
| Project root | decided | Work occurs in `/home/hanjinwei/p1/project/diffusion_policy`, the server-side Git checkout. |
| Training feedback construction | pending audit | Must be reconciled with the repository's actual policy/training interfaces before implementation. |
| Gate condition representation | pending audit and user confirmation | The method permits a compact existing global condition, but the repository architecture must be inspected first. |
| First perturbation mode | pending user confirmation | The specification permits mask or Gaussian; this research-affecting choice requires confirmation before Phase 3. |
| Checkpoint compatibility strategy | decided | User selected direct action-feedback channel concatenation. SC input is `2D`, output stays `D`, and old checkpoint compatibility is intentionally not retained. |
