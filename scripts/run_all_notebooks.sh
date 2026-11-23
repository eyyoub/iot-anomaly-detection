#!/usr/bin/env bash
# Execute the revision notebooks in order, saving outputs inside each notebook.
# Run inside tmux on the pod so a dropped SSH connection does not stop it:
#   tmux new -s rev 'bash scripts/run_all_notebooks.sh 2>&1 | tee run_all.log'
# Usage: bash scripts/run_all_notebooks.sh [R01 R02 ...]   (default: all)
set -euo pipefail
NBS=${@:-R01 R02 R03 R04 R05 R06}
for nb in $NBS; do
  f=$(ls ${nb}_*.ipynb)
  echo "=== $(date +%H:%M:%S) running $f"
  jupyter nbconvert --to notebook --execute --inplace "$f" \
      --ExecutePreprocessor.timeout=-1
done
echo "=== $(date +%H:%M:%S) all done"
