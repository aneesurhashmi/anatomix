#!/bin/bash
#BSUB -J apm_test
#BSUB -P your_slurm_account
#BSUB -n 36
#BSUB -q gpu
#BSUB -gpu num=1:j_exclusive=yes
#BSUB -R h100nvl
#BSUB -m lg05e15
#BSUB -W 144:00
#BSUB -u your.email@institution.edu
#BSUB -B
#BSUB -N
#BSUB -R rusage[mem=6000]
#BSUB -R span[hosts=1]
#BSUB -oo ./bash_logs/apm_test_%J.stdout
#BSUB -eo ./bash_logs/apm_test_%J.stderr
#BSUB -L /bin/bash


ml load anaconda3/2024.06
source activate /path/to/your/conda/env/

cd /path/to/anatomix
nvidia-smi


JOBID=$LSB_JOBID
HOST=$(hostname)
START=$(date)

python ./src/rag/create_rag_db.py
python train_apm.py --config ./configs/anatomix_config.yaml --mode inference --data.batch_size 48

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