#!/usr/bin/env bash
# save_csv_summary.sh
# Usage:  chmod +x save_csv_summary.sh && ./save_csv_summary.sh
# Optional: ./save_csv_summary.sh --with-png  (adds PNG paths)

set -euo pipefail
OUT_FILE="artifact_summary.txt"
LIST_PNG=false
[[ "${1:-}" == "--with-png" ]] && LIST_PNG=true

exec >"$OUT_FILE"

echo "=== Metrics CSVs (LR×σ, thresh 0.3) ==="
find tuning_runs/phasenet_Iquique -type f -name '*_metrics_thresh0.3.csv' | sort
echo

echo "=== Threshold-sweep CSVs (0.20–0.45) ==="
find sweep_thr_* -type f -name '*_metrics_thresh*.csv' | sort
echo

# Quick preview of each metrics CSV (first three lines)
echo "=== CSV previews (first 3 lines) ==="
while read -r csv; do
  echo "--- ${csv} ---"
  head -n 3 "${csv}"
  echo
done < <(find tuning_runs/phasenet_Iquique sweep_thr_* baseline_results \
         -type f -name '*_metrics_thresh*.csv' | sort)

if $LIST_PNG; then
  echo "=== PR-curve PNGs (only listed if --with-png) ==="
  find tuning_runs/phasenet_Iquique -type f -name '*_PR_curves.png' | sort
  find sweep_thr_* -type f -name '*_PR_curves.png' | sort
  echo
fi

echo "Summary saved to ${OUT_FILE}"
