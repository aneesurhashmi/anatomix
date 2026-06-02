#!/bin/bash
#BSUB -J data
#BSUB -P your_slurm_account
#BSUB -n 6
##BSUB -q gpu
##BSUB -gpu num=1
##BSUB -gpu num=1:j_exclusive=yes
##BSUB -R h100nvl
#BSUB -W 24:00
#BSUB -u your.email@institution.edu
#BSUB -B
#BSUB -N
#BSUB -R rusage[mem=64000]
#BSUB -R span[hosts=1]
#BSUB -oo ./bash_logs/data_%J.stdout
#BSUB -eo ./bash_logs/data_%J.stderr
#BSUB -L /bin/bash

ml load anaconda3/2024.06
source activate /path/to/your/conda/env/

JOBID=$LSB_JOBID
HOST=$(hostname)
START=$(date)


cd /path/to/anatomix/src/data/preprocess/

# python process_chest_imagenome.py --replace sg
# python data_split.py
python process_instruction_tuning.py

# STATUS=$?
# END=$(date)

# mail -s "LSF Job $JOBID Finished (status=$STATUS)" your.email@institution.edu <<EOF
# Job ID: $JOBID
# Job Name: $LSB_JOBNAME
# Queue: $LSB_QUEUE
# User: $LSB_SUB_USER
# Host: $HOST

# Start Time: $START
# End Time:   $END
# Exit Code:  $STATUS

# Output file: job.$JOBID.out
# Error file:  job.$JOBID.err
# EOF





# curl -X POST -H 'Content-type: application/json' \
#      --data "{\"text\": \"Job ${LSB_JOBID} finished with status ${STATUS}\"}" \
#      "$SLACK_WEBHOOK_URL"


#!/bin/bash
