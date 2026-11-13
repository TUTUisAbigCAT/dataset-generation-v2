#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --job-name=v2-sweep-v3
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/v2_sweep_v3_%A_%a.txt
# Deduplicated dropout sweep (configs/sweep_v3.json): five unique settings on
# [-1, 1] normalization, one array task each. Run from the repository folder:
#   sbatch submit_sweep_v3.sh                         # all five, 50 epochs
#   sbatch --array=3,4 submit_sweep_v3.sh             # selected rows only
#   sbatch --export=ALL,EPOCHS=30 submit_sweep_v3.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export PYTHONFAULTHANDLER=1

EPOCHS="${EPOCHS:-50}"
GRID="${GRID:-configs/sweep_v3.json}"
ID="${SLURM_ARRAY_TASK_ID}"
ROOT_OUT="/home/yxtao/links/scratch/dataset-generation-v2/runs/sweep_v3_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$ROOT_OUT"
CONFIG="${ROOT_OUT}/config_${ID}.json"
python -B scripts/sweep_config.py --grid "$GRID" --index "$ID" --output "$CONFIG"
echo "epochs=${EPOCHS} grid=${GRID} id=${ID} config=${CONFIG}"

srun python -B -u scripts/run_dreamt.py \
  --config "$CONFIG" --output-dir "${ROOT_OUT}/config_${ID}" \
  --epochs "$EPOCHS" --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"
