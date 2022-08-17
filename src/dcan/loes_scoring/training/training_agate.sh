#!/bin/bash -l
#SBATCH --job-name=loes-scoring.training.agate
#SBATCH --time=24:00:00
#SBATCH --partition=a100-4
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=loes-scoring.training.agate-%j.out
#SBATCH --error=loes-scoring.training.agate-%j.err

pwd; hostname; date
echo jobid=${SLURM_JOB_ID}; echo nodelist=${SLURM_JOB_NODELIST}

module load python3/3.8.3_anaconda2020.07_mamba
__conda_setup="$(`which conda` 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

conda activate /panfs/roc/groups/4/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/torch-env

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src":"/home/miran045/reine097/projects/AlexNet_Abrol2021/reprex"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/loes_scoring/training/training.py --batch-size=1 --epochs=1024 \
  --num-workers=6

echo COMPLETE
