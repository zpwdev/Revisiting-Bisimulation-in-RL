export HYDRA_FULL_ERROR=1

TASK_name=cheetah_run
EXP_NAME=${TASK_name}

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=0 nohup python -u train.py agent=simsr_sa  task=${TASK_name} exp_name=${EXP_NAME} >> ${TASK_name}.log &