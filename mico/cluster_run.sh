TASKS=(cheetah_run walker_run quadruped_run quadruped_walk finger_turn_hard acrobot_swingup reacher_hard reach_duplo hopper_hop)
SEEDS=(1 2 3 4 5 6)

for task in ${TASKS[@]}
do
  for seed in ${SEEDS[@]}
  do
  echo "#!/bin/bash" >> temprun.sh
  echo "#SBATCH --job-name=MICo_${task}_${seed}" >> temprun.sh
  echo "#SBATCH --cpus-per-task=10"  >> temprun.sh   # ask for 4 CPUs
  echo "#SBATCH --gres=gpu:1" >> temprun.sh         # ask for 1 GPU
  echo "#SBATCH --mem=90G" >> temprun.sh            # ask for 48 GB RAM
  # echo "#SBATCH --time=41:59:00" >> temprun.sh
  echo "#SBATCH -o /home/*/MICo/results/${task}_${seed}.log" >> temprun.sh
  echo "module load miniconda/3" >> temprun.sh
  echo "conda activate mico" >> temprun.sh
  echo "export HYDRA_FULL_ERROR=1" >> temprun.sh
  echo "CUDA_VISIBLE_DEVICES=0 python train.py agent=simsr_sa task=${task} exp_name=${task} seed=${seed}" >> temprun.sh
  eval "sbatch temprun.sh"
  rm temprun.sh
  done
done