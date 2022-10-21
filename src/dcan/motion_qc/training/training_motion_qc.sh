#!/bin/sh

#SBATCH --job-name=motion-training-alex-net # job name

#SBATCH --mem=90g        # memory per cpu-core (what is the default?)
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)
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
  --tb-prefix="MRIMotionQcScore_bcp_and_elabe" --epochs=1 --model="AlexNet3D_Dropout_Regression" \
  --dset="MRIMotionQcScoreDataset" \
  --qc_with_paths_csv='/panfs/jay/groups/6/faird/shared/projects/motion-QC-generalization/code/bcp_and_elabe_qc_train_space-infant_unique.csv' \
  --model-save-location="/home/feczk001/shared/data/AlexNet/motion-qc-model04-new.pt" \
  "MRIMotionQcScore_eLabe"
