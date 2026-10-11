#!/usr/bin/env bash
set -euo pipefail

# Run from the diff3 repository root. Requires prepared Push-T image data.
# Usage: bash run_tm_dp.sh vanilla 0
# Core modes: vanilla, tsm, me, full; extra ablations: me_only, fusion_only, me_fusion
MODE="${1:?Choose vanilla|tsm|me|full|me_only|fusion_only|me_fusion}"
GPU="${2:-0}"
case "$MODE" in
  vanilla)     TSM=false; ME=false; FUSION=false ;;
  tsm)         TSM=true;  ME=false; FUSION=false ;;
  me)          TSM=false; ME=true;  FUSION=true  ;;
  full)        TSM=true;  ME=true;  FUSION=true  ;;
  me_only)     TSM=false; ME=true;  FUSION=false ;;
  fusion_only) TSM=false; ME=false; FUSION=true  ;;
  me_fusion)   TSM=false; ME=true;  FUSION=true  ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac

# No random per-frame crops: preserves spatial correspondence for motion delta.
# Keep this setting identical in all four arms, including vanilla.
exec env CUDA_VISIBLE_DEVICES="$GPU" /home/hanjinwei/miniconda3/envs/robodiff/bin/python train.py \
  --config-name=train_diffusion_unet_hybrid_workspace \
  task=pusht_image \
  policy._target_=diffusion_policy.policy.tm_diffusion_unet_hybrid_image_policy.TMDiffusionUnetHybridImagePolicy \
  +policy.use_tsm="$TSM" \
  +policy.use_motion_excitation="$ME" \
  +policy.use_motion_condition_fusion="$FUSION" \
  policy.crop_shape=null \
  n_obs_steps=2 \
  training.device="cuda:0" \
  training.seed=42 \
  exp_name="tm_dp_${MODE}" \
  logging.mode=offline \
  "hydra.run.dir=data/outputs/tm_dp/${MODE}"
