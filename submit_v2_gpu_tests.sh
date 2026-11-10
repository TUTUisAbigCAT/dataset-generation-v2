#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2-gpu-tests
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/v2_gpu_tests_%j.txt
# Run from /project/def-plago/yxtao/dataset-generation-v2:  sbatch submit_v2_gpu_tests.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
module load scipy-stack/2024a   # numpy 1.26.4 / pandas / matplotlib come from this module
source ./.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export PYTHONFAULTHANDLER=1

python -c "import sys,torch,numpy;print('env',sys.version.split()[0],torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),numpy.__version__,torch.cuda.get_device_name(0))"

# $PROJECT is read-only inside Trillium jobs, and several tests write temporary
# files under the repository (work/). Run the suite from a copy on the job RAM disk.
export TMPDIR="${SLURM_TMPDIR}"
COPY="${SLURM_TMPDIR}/dataset-generation-v2"
mkdir -p "${COPY}"
tar -C . --exclude=./.venv --exclude=./.git -cf - . | tar -C "${COPY}" -xf -
chmod -R u+w "${COPY}"
cd "${COPY}"

python -B scripts/check_cuda.py
python -B -m unittest discover -s tests -v
echo "All tests finished"
