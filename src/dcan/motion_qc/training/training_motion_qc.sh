#!/bin/sh

#SBATCH --job-name=motion-training-alex-net # job name

#SBATCH --mem=90g        # memory per cpu-core (what is the default?)
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e motion-alex-net-training-%j.err
#SBATCH -o motion-alex-net-training-%j.out

#SBATCH -A feczk001

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/motion_qc/training/training.py --num-workers=4 --batch-size=8 \
  --tb-prefix="MRIMotionQcScore_BCP_and_eLabe" --epochs=64 --model="AlexNet" --dset="MRIMotionQcScoreDataset" \
  --qc_with_paths_csv='/home/miran045/reine097/projects/AlexNet_Abrol2021/data/BCP_and_eLabe/bcp_and_elabe_qc_test.csv' \
  "MRIMotionQcScore_eLabe_and_BCP"
