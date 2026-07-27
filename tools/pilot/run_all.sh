#!/usr/bin/env bash
# NONFORMAL_DIAGNOSTIC (read-only). Reproduces PILOT_LOG.md entry 001.
# Not Docker-bound, not receipt-signed, writes only to /tmp. See README.md.
set -euo pipefail

PY="${PILOT_PYTHON:-/home/wxs/anaconda3/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ORDER=(
  d01_probe_hand_fields.py
  d02_action_is_robot_command.py
  d03_gripper_leakage.py
  d04_calibration_global.py
  d05_calibration_points.py
  d06_pairing_control.py
  d07_delta_calibration.py
  d08_delta_magnitudes.py
  d09_variant_ab_retrieval.py
  d10_residual_analysis.py
  d11_timebase.py
  d12_oracle_ceiling.py
  d13_ablation_cache.py
  d15_ablation_run.py
  d14_ablation_pixel.py
  d16_ablation_run_pixel.py
)

for script in "${ORDER[@]}"; do
  echo
  echo "################################################################"
  echo "# $script"
  echo "################################################################"
  "$PY" "$HERE/$script"
done

echo
echo "pilot diagnostics complete. Intermediates left in /tmp (h2r_*.npy, h2r_*.pkl)."
