# Progress log

## 2026-09-21 — initialization

- Created persistent planning files for the server-side project root.
- Phase 0 started; no policy, configuration, or training behavior has been changed.
- Located the server checkout and recorded its current commit and pre-existing dirty state. No reset, cleanup, or code change was performed.
- Audited the low-dimensional policy, conditional UNet, workspace, checkpoint behavior, EMA behavior, and default low-dimensional configuration. No task rollout or policy edit has been performed.
- Applied the authorized cleanup of historical base-policy changes and outputs while preserving the repository's unrelated baseline source files. The workpoint-one implementation has not begun.
- Phase 0 complete. Ran the actual low-dimensional `compute_loss` path and a two-step real denoising loop in `robodiff` on GPU: finite loss `1.3082119226455688`, action input/output `[1,16,2]`, observation `[1,16,20]`, execution slice `[1,8,2]`, and `65,783,430` parameters. This is an interface check, not a task experiment.
- Removed the temporary Phase 0 check script after recording the result; it will not be committed.
- User authorized Phase 1 direct channel concatenation. The planned first benchmark is RoboMimic Can low-dimensional; its configured HDF5 dataset is not present on the server, so no training or download has started.
- Phase 1 configuration initially used a nested conditional arithmetic resolver that Hydra rejected while resolving `policy.model.input_dim`. Replaced it with a separately resolved `base_input_dim` followed by a quoted arithmetic resolver; no policy behavior or method decision changed.
- Phase 1 complete. `policy.use_self_condition=false` resolves Can dimensions to input/output `7/7`; enabled resolves to `14/7`. GPU interface result: finite loss, detached feedback `[1,16,7]`, finite chunk `[1,16,7]`, executed slice `[1,8,7]`. Temporary check script removed after recording this result.
- Published Phase 1 to `git@github.com:Lin-zh-handsome/diff2.git` on `main`. Method commit: `37bcdfe feat: add action self-conditioning baseline`; target repository had an unrelated README-only initial commit, so its merge commit is `90d6f5d`. Historical output deletions were not included.
- Updated SC training to the requested Analog Bits 50% self-conditioning / 50% zero-feedback regime at the same sampled timestep. Verified Python syntax with `py_compile`; no test or training run was added.
