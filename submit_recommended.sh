#!/bin/bash
#SBATCH --account=def-plago       # 换成你的账户名 (def-xxxx)
#SBATCH --cpus-per-task=4         # CPU 数量
#SBATCH --gpus-per-node=1         # GPU 数量
#SBATCH --time=24:00:00           # 100 epochs; 按实际单轮耗时调整
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/recommended_%j.txt
# Recommended retrain (docs/代码修改指南.md): EMA 0.9999 + masked missing values +
# S4 dropout 0.03 + no input dropout + per-epoch checkpoints, 100 epochs.
set -eo pipefail

module load StdEnv/2023  gcc/12.3
module load opencv/4.9.0
module load python/3.11

source ./.venv/bin/activate
python scripts/run_dreamt.py --config configs/dreamt_recommended.json --epochs 100 --batch-size 8 --threads 4
