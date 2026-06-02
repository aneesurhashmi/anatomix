#!/bin/bash
#BSUB -J lm_training
#BSUB -P your_slurm_account
#BSUB -n 32
#BSUB -q gpu
#BSUB -gpu num=4:j_exclusive=yes
#BSUB -R h100nvl
#BSUB -W 144:00
##BSUB -m lg02e09
#BSUB -u your.email@institution.edu
#BSUB -B
#BSUB -N
#BSUB -R rusage[mem=32000]
#BSUB -R span[hosts=1]
#BSUB -oo ./bash_logs/lm_training_%J.stdout
#BSUB -eo ./bash_logs/lm_training_%J.stderr
#BSUB -L /bin/bash


ml load anaconda3/2024.06
source activate /path/to/your/conda/env/

cd /path/to/anatomix
nvidia-smi


JOBID=$LSB_JOBID
HOST=$(hostname)
START=$(date)

# echo "Create the RAD DB first"
# python ./src/rag/create_rag_db.py

echo "Running LM Training sciprt"

# # training step 1
# torchrun --nproc_per_node=4  train_lm.py --config ./configs/anatomix_config.yaml \
#                     --lm.model_args.training_step 1 \
#                     --lm.sft_config.num_train_epochs 3 \
#                     --lm.sft_config.per_device_train_batch_size 2 \
#                     --lm.sft_config.per_device_eval_batch_size 2 \
#                     --lm.sft_config.gradient_accumulation_steps 4 

# training step 2
torchrun --nproc_per_node=4 train_lm.py --config ./configs/anatomix_config.yaml \
                    --lm.model_args.training_step 2 \
                    --lm.sft_config.num_train_epochs 4 \
                    --lm.sft_config.per_device_train_batch_size 1 \
                    --lm.sft_config.per_device_eval_batch_size 1 \
                    --lm.sft_config.gradient_accumulation_steps 8 \
                    --lm.sft_config.dataloader_num_workers 24




STATUS=$?
END=$(date)

mail -s "LSF Job $JOBID Finished (status=$STATUS)" your.email@institution.edu <<EOF
Job ID: $JOBID
Job Name: $LSB_JOBNAME
Queue: $LSB_QUEUE
User: $LSB_SUB_USER
Host: $HOST

Start Time: $START
End Time:   $END
Exit Code:  $STATUS

Output file: job.$JOBID.out
Error file:  job.$JOBID.err
EOF