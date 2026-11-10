#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=v2-recommended
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/v2_recommended_%j.txt
# Run from /project/def-plago/yxtao/dataset-generation-v2:
#   sbatch submit_v2_recommended.sh                          # 50 epochs (default)
#   sbatch --export=ALL,EPOCHS=100 submit_v2_recommended.sh  # only if 100 epochs fit in 24 h
# Trillium accepts jobs of at most 24 h and run_dreamt.py cannot resume, so all
# epochs (plus final evaluation/generation) must finish inside this one job.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
module load scipy-stack/2024a   # numpy 1.26.4 / pandas / matplotlib come from this module
source ./.venv/bin/activate

# Set before Python imports NumPy/PyTorch (same thread policy as the earlier runs).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export PYTHONFAULTHANDLER=1

EPOCHS="${EPOCHS:-50}"
CONFIG=configs/dreamt_recommended.json   # EMA 0.9999 + masked + S4 dropout 0.03 + per-epoch checkpoints
OUTPUT="/home/yxtao/links/scratch/dataset-generation-v2/runs/recommended_${SLURM_JOB_ID}"   # scratch: writable in jobs

python -c "import sys,torch,numpy;print('env',sys.version.split()[0],torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),numpy.__version__,torch.cuda.get_device_name(0))"
echo "epochs=${EPOCHS} config=${CONFIG} output=${OUTPUT}"

srun python -B -u scripts/run_dreamt.py \
  --config "$CONFIG" --output-dir "$OUTPUT" \
  --epochs "$EPOCHS" --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed recommended run (%s epochs). Outputs: %s\n' "$EPOCHS" "$OUTPUT"
