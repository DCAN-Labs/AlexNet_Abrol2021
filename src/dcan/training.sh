#!/bin/sh

#SBATCH --job-name=p2ch11.training
#SBATCH --mem=90g
#SBATCH --time=24:00:00
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=6

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e alex-net-training-%j.err
#SBATCH -o alex-net-training-%j.out

#SBATCH -A rando149

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/training.py --num-workers=1 --batch-size=1 \
  --tb-prefix="age_regression" --epochs=1 --model="AlexNet" "dcan"
